"""Promote explicitly approved, complete records into a grouped training manifest.

python -m tools.promote_review_dataset --manifest DATASET/manifest.json --reviews DATASET/review.csv --output DATASET/manifest.reviewed.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from ureter_semis.labels import read_class_mask

from .prepare_review_dataset import assign_groups, sha256


def promote(manifest_path: str | Path, reviews_path: str | Path, output_path: str | Path,
            labeled_fraction: float = 0.2, seed: int = 42) -> Path:
    if not 0 < labeled_fraction < 1:
        raise ValueError("labeled_fraction must be between 0 and 1")
    manifest_path, reviews_path, output = map(lambda x: Path(x).resolve(),
                                             (manifest_path, reviews_path, output_path))
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing manifest: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2 or "draft_samples" not in manifest:
        raise ValueError("Expected a version-2 review draft")
    root = Path(manifest["root"])
    by_id = {r["sample_id"]: r for r in manifest["draft_samples"]}
    with reviews_path.open(encoding="utf-8-sig", newline="") as handle:
        decisions = list(csv.DictReader(handle))
    if len({d["sample_id"] for d in decisions}) != len(decisions):
        raise ValueError("Duplicate review decisions")
    approved = []
    for decision in decisions:
        sid = decision["sample_id"]
        if sid not in by_id:
            raise ValueError(f"Unknown review sample: {sid}")
        status = decision.get("review_status", "").strip()
        if status not in {"pending", "approved", "rejected"}:
            raise ValueError(f"Invalid review status: {sid}")
        if status != "approved":
            continue
        if not decision.get("reviewer", "").strip():
            raise ValueError(f"Approved sample needs reviewer identity: {sid}")
        if decision.get("source_group_verified", "").strip().lower() != "true":
            raise ValueError(f"Verified source group required: {sid}")
        patient, video = decision.get("patient_id", "").strip(), decision.get("video_id", "").strip()
        if not patient and not video:
            raise ValueError(f"Patient or original-video identity required: {sid}")
        if decision.get("fiber_annotation_completeness") != "complete":
            raise ValueError(f"Full fiber annotation review required (including true absence): {sid}")
        presence = decision.get("stone_presence_review")
        if presence not in {"present", "absent"}:
            raise ValueError(f"Stone presence must be explicitly reviewed: {sid}")
        record = dict(by_id[sid])
        if sha256(root / record["image"]) != record["image_sha256"]:
            raise ValueError(f"Source image changed after audit: {sid}")
        corrected = decision.get("corrected_mask", "").strip()
        mask_path = (reviews_path.parent / corrected).resolve() if corrected else root / record["mask"]
        if not corrected and sha256(mask_path) != record["mask_sha256"]:
            raise ValueError(f"Mask changed without an explicit correction record: {sid}")
        mask = read_class_mask(mask_path)
        if mask.shape != (record["height"], record["width"]) or not set(np.unique(mask)).issubset({0,1,2,255}):
            raise ValueError(f"Invalid reviewed mask: {sid}")
        if not (mask != 255).any() or bool((mask == 1).any()) != (presence == "present"):
            raise ValueError(f"Reviewed presence disagrees with mask: {sid}")
        semantic_hash = hashlib.sha256(str(mask.shape).encode()+mask.tobytes()).hexdigest()
        record.update({"mask": str(mask_path), "mask_sha256": sha256(mask_path), "mask_semantic_sha256": semantic_hash,
                       "patient_id": patient or None,
                       "video_id": video or None, "reviewer": decision["reviewer"].strip(),
                       "review_note": decision.get("review_note", ""), "review_status": "approved",
                       "stone_presence_review": presence, "fiber_annotation_completeness": "complete",
                       "source_group_verified": True, "training_eligible": True})
        approved.append(record)
    if not approved:
        raise ValueError("No explicitly approved records; draft remains untrainable")
    # Union verified patients AND videos before deduplicating: missing patient
    # metadata on one frame cannot split another frame from the same video.
    assign_groups(approved, seed, [0.7, 0.15, 0.15])
    duplicate_groups = defaultdict(list)
    for row in approved:
        duplicate_groups[row["image_sha256"]].append(row)
    unique = []
    for rows in duplicate_groups.values():
        if len({r["mask_semantic_sha256"] for r in rows}) > 1:
            raise ValueError("Reviewed duplicate image has conflicting masks; resolve before promotion")
        canonical = min(rows, key=lambda r: r["sample_id"])
        canonical["equivalent_sample_ids"] = [r["sample_id"] for r in rows]
        unique.append(canonical)
    splits = {s: sorted([r for r in unique if r["proposed_split"] == s], key=lambda r:r["sample_id"])
              for s in ("train", "val", "test")}
    if any(not rows for rows in splits.values()) or len(splits["train"]) < 3:
        raise ValueError("Insufficient reviewed groups for nonempty train/val/test; add verified reviews")
    if not any(r["stone_presence_review"] == "absent" for r in splits["train"]):
        raise ValueError("Reviewed training split needs real stone-negative images")
    if not any(r["stone_presence_review"] == "present" for r in splits["train"]):
        raise ValueError("Reviewed training split needs stone-positive images")
    for split in ("val", "test"):
        if {r["stone_presence_review"] for r in splits[split]} != {"present", "absent"}:
            raise ValueError(f"{split} needs reviewed positive and negative images")
    rng = random.Random(seed)
    train = splits["train"][:]
    rng.shuffle(train)
    # Ensure both signs of stone presence are represented in supervised data.
    chosen = [next(r for r in train if r["stone_presence_review"] == status) for status in ("present", "absent")]
    count = min(len(train)-1, max(2, round(len(train)*labeled_fraction)))
    chosen_ids = {r["sample_id"] for r in chosen}
    chosen += [r for r in train if r["sample_id"] not in chosen_ids][:count-2]
    chosen_ids = {r["sample_id"] for r in chosen}
    result = {k:v for k,v in manifest.items() if k not in {"draft_samples", "train", "val", "test", "training_blockers"}}
    result.update({"role": "reviewed_grouped_dataset", "training_ready": True,
                   "train": {"labeled": chosen, "unlabeled": [r for r in train if r["sample_id"] not in chosen_ids]},
                   "val": splits["val"], "test": splits["test"],
                   "review_provenance": {"draft_manifest_sha256": sha256(manifest_path),
                                         "reviews_sha256": sha256(reviews_path), "approved_records": len(approved),
                                         "unique_images": len(unique), "seed": seed,
                                         "all_records_have_patient_id": all(r.get("patient_id") for r in unique),
                                         "scope": "reviewed patient/video groups; training-source overlap still requires provenance review"}})
    result["split_policy"] = {"seed": seed, "ratios": [0.7,0.15,0.15],
                              "method": "union verified patient/video, filename families and exact image hashes before split",
                              "patient_independent_verified": all(r.get("patient_id") for r in unique)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labeled-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(promote(args.manifest, args.reviews, args.output, args.labeled_fraction, args.seed))


if __name__ == "__main__":
    main()
