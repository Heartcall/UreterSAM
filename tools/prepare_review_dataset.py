"""Build a provenance-preserving, review-gated YOLO dataset draft.

python -m tools.prepare_review_dataset --config configs/dataset_sources_v2.json --output datasets/ureter_semis_v2_review_20261027
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw
import yaml

from .convert_dataset import IMAGE_SUFFIXES, _target_class, read_yolo_sample


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    encoded.tofile(str(path))


def grouping_hint(source_id: str, stem: str) -> tuple[str, str]:
    family = stem.split(".rf.")[0]
    match = re.fullmatch(r"(.+?)(?:_mp4|\.mp4)[-_](\d+)(?:_jpg)?", family, re.IGNORECASE)
    if match:
        # Frame numbers and video identities are hints, not verified timestamps.
        return f"{source_id}:video_hint:{match.group(1)}", "filename_video_hint_unverified"
    return f"{source_id}:family:{family}", "augmentation_family_only"


def assign_groups(records: list[dict], seed: int, ratios: list[float]) -> None:
    """Union filename groups and exact duplicates before assigning any split."""
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[find(b)] = find(a)
    seen_group, seen_hash = {}, {}
    for i, row in enumerate(records):
        group_keys = [row["group_hint"]]
        for field in ("patient_id", "video_id"):
            if row.get(field):
                group_keys.append(f"verified_{field}:{row[field]}")
        for key, seen in [(key, seen_group) for key in group_keys]+[(row["image_sha256"], seen_hash)]:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    for members in groups.values():
        canonical = min(records[i]["sample_id"] for i in members)
        token = hashlib.sha256(f"{seed}:{canonical}".encode()).hexdigest()
        value = int(token[:16], 16) / 2**64
        split = "train" if value < ratios[0] else "val" if value < ratios[0]+ratios[1] else "test"
        for i in members:
            records[i]["proposed_split"] = split
            records[i]["group_id"] = token[:20]


def contact_sheets(records: list[dict], output: Path, name: str) -> list[str]:
    directory = output / "review" / name
    directory.mkdir(parents=True, exist_ok=True)
    pages = []
    for start in range(0, len(records), 48):
        rows = records[start:start+48]
        canvas = Image.new("RGB", (6*192, 8*180), "#202020")
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(rows):
            x, y = (i % 6)*192, (i // 6)*180
            with Image.open(output / row["image"]) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((188, 138))
                canvas.paste(thumb, (x+(192-thumb.width)//2, y))
            draw.text((x+4, y+140), f"Review {row['review_index']:04d}", fill="white")
            draw.text((x+4, y+155), f"S:{int(row['has_stone_annotation'])} F:{int(row['has_fiber_annotation'])}  UNREVIEWED", fill="#ffd080")
        path = directory / f"page_{start//48:02d}.jpg"
        canvas.save(path, quality=90)
        pages.append(path.relative_to(output).as_posix())
    return pages


def build_review_dataset(config_path: str | Path, output_path: str | Path) -> Path:
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(output_path).resolve()
    ratios = config.get("split_ratios", [0.7, 0.15, 0.15])
    if len(ratios) != 3 or any(v <= 0 for v in ratios) or not np.isclose(sum(ratios), 1):
        raise ValueError("Three positive split ratios must sum to one")
    sources = config["sources"]
    ids = [source["id"] for source in sources]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", s) for s in ids):
        raise ValueError("Source ids must be unique safe identifiers")
    # Validate every source before producing output.
    for source in sources:
        root = Path(source["path"]).resolve()
        yml = root / "data.yaml"
        data = yaml.safe_load(yml.read_text(encoding="utf-8"))
        names = data["names"]
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names, key=int)]
        if names != source["expected_names"]:
            raise ValueError(f"Source categories changed: {root}")
        source["resolved_root"] = str(root)
        source["yaml_sha256"] = sha256(yml)
        source["roboflow"] = data.get("roboflow")
        for name in names:
            target = _target_class(name)
            action = source["class_policy"][name]
            if (target == 1 and action != "stone") or (target == 2 and action != "fiber"):
                raise ValueError(f"Source policy disagrees with conversion for {name}")
            if target is None and action not in {"background", "auxiliary_laser_dot", "unknown_ignore_255"}:
                raise ValueError(f"Unsupported policy for {name}")
    output.mkdir(parents=True, exist_ok=False)
    records, image_hashes, counters = [], defaultdict(list), {}
    for source in sources:
        root = Path(source["resolved_root"])
        names = dict(enumerate(source["expected_names"]))
        counter = Counter()
        for split in ("train", "valid", "val", "test"):
            image_dir = root / split / "images"
            if not image_dir.is_dir():
                continue
            if split == "val" and (root / "valid" / "images").is_dir():
                continue
            for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
                label_path = root / split / "labels" / f"{image_path.stem}.txt"
                mask, metadata, auxiliary = read_yolo_sample(image_path, label_path, names, mark_unknown=True)
                digest = sha256(image_path)
                # Include split and filename hash: no silent same-name overwrite.
                sid = f"{source['id']}__{split}__{hashlib.sha256(image_path.name.encode()).hexdigest()[:20]}"
                image_rel = f"images/{sid}{image_path.suffix.lower()}"
                mask_rel = f"masks/{sid}.png"
                (output / "images").mkdir(exist_ok=True)
                shutil.copy2(image_path, output / image_rel)
                write_png(output / mask_rel, mask)
                auxiliary_files = {}
                for role, array in auxiliary.items():
                    path = f"auxiliary/{role}/{sid}.png"
                    write_png(output / path, array)
                    auxiliary_files[role] = path
                hint, quality = grouping_hint(source["id"], image_path.stem)
                record = {"review_index": len(records), "sample_id": sid, "image": image_rel,
                          "mask": mask_rel, "source_dataset": source["id"], "source_image": str(image_path),
                          "source_label": str(label_path), "source_split": split,
                          "source_label_sha256": sha256(label_path) if label_path.is_file() else None,
                          "image_sha256": digest, "mask_sha256": sha256(output / mask_rel),
                          "auxiliary_masks": auxiliary_files, "group_hint": hint,
                          "grouping_evidence": quality, "source_group_verified": False,
                          "patient_id": None, "video_id": None, "sequence_verified": False,
                          "frame_index": None, "timestamp_s": None, "review_status": "pending",
                          "reviewer": "", "review_note": "", "corrected_mask": "",
                          "stone_presence_review": "unknown", "fiber_annotation_completeness": "unknown",
                          "training_eligible": False, **metadata}
                records.append(record)
                image_hashes[digest].append(record)
                counter["images"] += 1
                counter["stone_annotation_images"] += int(metadata["has_stone_annotation"])
                counter["fiber_annotation_images"] += int(metadata["has_fiber_annotation"])
                for name, value in metadata["annotation_counts"].items():
                    counter[f"instances:{name}"] += value
                for flag in metadata["review_flags"]:
                    counter[f"flag:{flag}"] += 1
        counters[source["id"]] = dict(counter)
        print(f"Converted {source['id']}: {counter['images']} images", flush=True)
    if not records:
        raise ValueError("No dataset records")
    duplicate_groups = []
    for digest, group in image_hashes.items():
        if len(group) > 1:
            conflict = len({r["mask_sha256"] for r in group}) > 1
            flag = "duplicate_conflicting_masks" if conflict else "duplicate_image_requires_canonical_review"
            for row in group:
                row["review_flags"].append(flag)
            duplicate_groups.append({"sha256": digest, "sample_ids": [r["sample_id"] for r in group],
                                     "mask_conflict": conflict})
    assign_groups(records, int(config.get("split_seed", 42)), ratios)
    negative_candidates = [r for r in records if not r["has_stone_annotation"] and r["label_file_present"]]
    laser_candidates = [r for r in records if "laser_spot_without_fiber_annotation" in r["review_flags"]]
    unknown_candidates = [r for r in records if any(f.startswith("unsupported_class:") for f in r["review_flags"])]
    pages = {"negative_candidates": contact_sheets(negative_candidates, output, "negative_candidates"),
             "laser_without_fiber": contact_sheets(laser_candidates, output, "laser_without_fiber"),
             "unknown_classes": contact_sheets(unknown_candidates, output, "unknown_classes")}
    manifest = {"schema_version": 2, "root": str(output), "classes": {"0": "background", "1": "stone", "2": "fiber"},
                "ignore_index": 255, "training_ready": False, "role": "unreviewed_development_draft",
                "training_blockers": ["Annotation completeness and negative candidates require review.",
                                      "Patient/video identities and filename-derived grouping are unverified."],
                "sources": sources, "source_config_sha256": sha256(config_path),
                "train": {"labeled": [], "unlabeled": []}, "val": [], "test": [], "draft_samples": records,
                "split_policy": {"seed": config.get("split_seed", 42), "ratios": ratios,
                                 "method": "union filename group hints + exact image hashes before stable-hash assignment",
                                 "independent_patient_evaluation_verified": False}}
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["review_index", "sample_id", "source_dataset", "source_image", "source_split", "proposed_split",
              "group_id", "grouping_evidence", "source_group_verified", "has_stone_annotation", "has_fiber_annotation",
              "review_status", "stone_presence_review", "fiber_annotation_completeness", "patient_id", "video_id",
              "reviewer", "review_note", "corrected_mask", "training_eligible", "review_flags"]
    with (output / "review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow({**row, "review_flags": ";".join(row["review_flags"])})
    summary = {"images": len(records), "sources": counters, "negative_candidates_not_verified": len(negative_candidates),
               "laser_without_fiber_annotation": len(laser_candidates), "unsupported_class_images": len(unknown_candidates),
               "duplicate_groups": duplicate_groups, "proposed_split_counts": dict(Counter(r["proposed_split"] for r in records)),
               "group_counts": len({r["group_id"] for r in records}), "review_pages": pages,
               "training_ready": False, "training_eligible_images": 0,
               "notes": ["No candidate has been automatically declared a true negative.",
                         "Filename group hints prevent likely leakage but do not prove patient independence.",
                         "Laser spot auxiliary masks do not overwrite stone masks."]}
    (output / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build_review_dataset(args.config, args.output))


if __name__ == "__main__":
    main()
