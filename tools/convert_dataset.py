from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _target_class(name: str) -> int | None:
    normalized = name.replace("_", " ").replace("-", " ").strip().lower()
    if normalized in {"stone", "kidney stone", "kidney stones", "stone new"}:
        return 1
    # A laser spot is not a physical fiber. Ambiguous source labels need review.
    if normalized in {"fiber", "laser fiber", "laser fibre", "fibre"}:
        return 2
    return None


def _write_sample(image_path: Path, mask: np.ndarray, output: Path, split: str, name_prefix: str = "") -> dict:
    image_dir = output / split / "images"
    mask_dir = output / split / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{name_prefix}__{image_path.stem}" if name_prefix else image_path.stem
    destination_image = image_dir / f"{stem}{image_path.suffix.lower()}"
    destination_mask = mask_dir / f"{stem}.png"
    if destination_image.exists() or destination_mask.exists():
        raise FileExistsError(f"Duplicate output name requires review: {stem}")
    shutil.copy2(image_path, destination_image)
    ok, encoded = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError(f"Cannot encode mask: {destination_mask}")
    encoded.tofile(str(destination_mask))
    return {"image": destination_image.relative_to(output).as_posix(), "mask": destination_mask.relative_to(output).as_posix()}


def _fill_polygon(mask: np.ndarray, polygon: np.ndarray, target: int) -> None:
    polygon_mask = np.zeros_like(mask)
    cv2.fillPoly(polygon_mask, [polygon], color=target)
    # MULTICLASS ADAPTATION: visible fiber wins over stone on overlap.
    if target == 2:
        mask[polygon_mask == 2] = 2
    else:
        mask[(polygon_mask == 1) & (mask == 0)] = 1


def read_yolo_sample(image_path: Path, label_path: Path, names: dict[int, str],
                     mark_unknown: bool = False) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    """Rasterize one export; preserve non-target polygons for data review.

    No label file is UNKNOWN, not evidence of a target-free image. Review-mode
    masks use 255 for missing labels and unsupported class regions.
    """
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {image_path}")
    height, width = image.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    auxiliary: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    flags = []
    if not label_path.is_file():
        flags.append("missing_label_file")
        if mark_unknown:
            mask[:] = 255
    else:
        for number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            values = line.split()
            if not values:
                continue
            if len(values) < 7 or (len(values)-1) % 2:
                raise ValueError(f"Invalid polygon at {label_path}:{number}")
            try:
                class_id = int(values[0])
                coordinates = np.asarray(values[1:], np.float32).reshape(-1, 2)
            except ValueError as exc:
                raise ValueError(f"Invalid numeric annotation at {label_path}:{number}") from exc
            if class_id not in names or not np.isfinite(coordinates).all():
                raise ValueError(f"Unknown class or nonfinite coordinates at {label_path}:{number}")
            if coordinates.min() < -1e-4 or coordinates.max() > 1.0001:
                raise ValueError(f"Out-of-range normalized polygon at {label_path}:{number}")
            name = names[class_id]
            counts[name] = counts.get(name, 0)+1
            coordinates = np.clip(coordinates, 0, 1)*np.array([width, height], np.float32)
            polygon = np.rint(coordinates).astype(np.int32)
            target = _target_class(name)
            if target is not None:
                _fill_polygon(mask, polygon, target)
            else:
                normalized = name.replace("-", " ").replace("_", " ").strip().lower()
                role = "laser_dot" if normalized in {"laser", "laser dot"} else "tissue" if normalized == "tissue" else "unknown"
                layer = auxiliary.setdefault(role, np.zeros_like(mask))
                cv2.fillPoly(layer, [polygon], color=1)
                if role == "unknown":
                    flags.append(f"unsupported_class:{name}")
    has_stone = any(_target_class(name) == 1 for name in counts)
    has_fiber = any(_target_class(name) == 2 for name in counts)
    if not has_stone:
        flags.append("no_stone_annotation_not_verified_negative")
    if not has_fiber:
        flags.append("fiber_absence_or_completeness_unverified")
    if "laser_dot" in auxiliary and not has_fiber:
        flags.append("laser_spot_without_fiber_annotation")
    if mark_unknown and "unknown" in auxiliary:
        mask[(auxiliary["unknown"] > 0) & (mask == 0)] = 255
    return mask, {"width": width, "height": height, "label_file_present": label_path.is_file(),
                  "annotation_counts": counts, "has_stone_annotation": has_stone,
                  "has_fiber_annotation": has_fiber, "review_flags": sorted(set(flags))}, auxiliary


def _convert_yolo(source: Path, output: Path, name_prefix: str = "",
                  seen_hashes: set[str] | None = None) -> dict[str, list[dict]]:
    yaml_path = next(iter(source.glob("*.yaml")), None)
    if yaml_path is None:
        raise FileNotFoundError(f"YOLO export has no data YAML in {source}")
    names_data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")).get("names", {})
    names = {i: str(name) for i, name in enumerate(names_data)} if isinstance(names_data, list) else {int(i): str(name) for i, name in names_data.items()}
    records: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for source_split, target_split in (("train", "train"), ("valid", "val"), ("val", "val"), ("test", "test")):
        image_dir, label_dir = source / source_split / "images", source / source_split / "labels"
        if not image_dir.is_dir() or (source_split == "val" and records["val"]):
            continue
        for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            if seen_hashes is not None:
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                if digest in seen_hashes:
                    raise ValueError(f"Duplicate image needs annotation/split review: {image_path}")
                seen_hashes.add(digest)
            label_path = label_dir / f"{image_path.stem}.txt"
            mask, metadata, _ = read_yolo_sample(image_path, label_path, names)
            if not metadata["label_file_present"]:
                raise ValueError(f"Missing label cannot be converted to background: {label_path}")
            sample = _write_sample(image_path, mask, output, target_split, name_prefix)
            sample["source_dataset"] = source.name
            sample["source_split"] = source_split
            records[target_split].append(sample)
    return records


def _find_coco_json(split_dir: Path, source: Path) -> Path | None:
    candidates = list(split_dir.glob("*.json")) + list(source.glob(f"*{split_dir.name}*.json"))
    return next((path for path in candidates if "annotation" in path.name.lower() or "coco" in path.name.lower()), None)


def _convert_coco(
    source: Path,
    output: Path,
    name_prefix: str = "",
    seen_hashes: set[str] | None = None,
) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    found = False
    for source_split, target_split in (("train", "train"), ("valid", "val"), ("val", "val"), ("test", "test")):
        split_dir = source / source_split
        annotation_path = _find_coco_json(split_dir, source)
        if annotation_path is None:
            continue
        found = True
        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}
        annotations: dict[int, list[dict]] = {}
        for annotation in coco.get("annotations", []):
            annotations.setdefault(int(annotation["image_id"]), []).append(annotation)
        for image_info in coco.get("images", []):
            filename = Path(image_info["file_name"])
            candidates = [split_dir / filename, split_dir / "images" / filename, source / filename]
            image_path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if image_path is None:
                continue
            if seen_hashes is not None:
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            for annotation in annotations.get(int(image_info["id"]), []):
                target = _target_class(categories.get(int(annotation.get("category_id", -1)), ""))
                segmentation = annotation.get("segmentation")
                if target is None or not isinstance(segmentation, list):
                    continue
                for coordinates in segmentation:
                    polygon = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
                    _fill_polygon(mask, np.rint(polygon).astype(np.int32), target)
            records[target_split].append(_write_sample(image_path, mask, output, target_split, name_prefix))
    if not found:
        raise FileNotFoundError(f"COCO export has no split annotation JSON in {source}")
    return records


def prepare_dataset(
    source: str | Path | Sequence[str | Path],
    output: str | Path,
    format: str = "auto",
    labeled_fraction: float = 0.2,
    seed: int = 42,
) -> Path:
    sources = [Path(source)] if isinstance(source, (str, Path)) else [Path(item) for item in source]
    sources = [item.expanduser().resolve() for item in sources]
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory; refusing to overwrite: {output}")
    if not 0 < labeled_fraction < 1:
        raise ValueError("labeled_fraction must be strictly between 0 and 1")
    if not sources:
        raise ValueError("at least one source directory is required")
    if format == "auto":
        detected = ["yolo" if next(iter(item.glob("*.yaml")), None) else "coco" for item in sources]
        if len(set(detected)) != 1:
            raise ValueError("all source directories must use the same annotation format")
        format = detected[0]
    records: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    seen_hashes: set[str] = set()
    for source_item in sources:
        name_prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", source_item.name).strip("_")
        converted = _convert_yolo(source_item, output, name_prefix, seen_hashes) if format == "yolo" else _convert_coco(source_item, output, name_prefix, seen_hashes) if format == "coco" else None
        if converted is None:
            raise ValueError("format must be auto, yolo, or coco")
        for split in records:
            records[split].extend(converted[split])
    if not records["train"] or not records["val"]:
        raise RuntimeError("The export must contain train and validation images")
    train_records = records["train"][:]
    random.Random(seed).shuffle(train_records)
    labeled_count = max(1, round(len(train_records) * labeled_fraction))
    manifest = {
        "root": str(output),
        "training_ready": False,
        "training_blockers": ["New conversions require annotation and split review; use the review-dataset workflow."],
        "classes": {"0": "background", "1": "stone", "2": "fiber"},
        "train": {"labeled": train_records[:labeled_count], "unlabeled": train_records[labeled_count:] or train_records[:labeled_count]},
        "val": records["val"],
        "test": records["test"],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YOLO/COCO polygons to 0/1/2 semantic PNG masks")
    parser.add_argument("--source", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=("auto", "yolo", "coco"), default="auto")
    parser.add_argument("--labeled-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(prepare_dataset(args.source, args.output, args.format, args.labeled_fraction, args.seed))


if __name__ == "__main__":
    main()
