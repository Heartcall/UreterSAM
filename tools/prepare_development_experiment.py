"""Build a user-authorized S2 experiment without inventing review or case provenance."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image

from ureter_semis.experiment import atomic_json, sha256
from ureter_semis.labels import read_class_mask


def build(draft_path: str | Path, reviewed_path: str | Path, session_path: str | Path,
          output_path: str | Path, seed: int = 42, labeled_fraction: float = 0.2) -> Path:
    if not 0 < labeled_fraction < 1:
        raise ValueError("labeled_fraction must lie between zero and one")
    draft_path, reviewed_path, session_path, output = map(lambda p: Path(p).resolve(),
                                                        (draft_path, reviewed_path, session_path, output_path))
    if output.exists():
        raise FileExistsError(output)
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
    session = json.loads(session_path.read_text(encoding="utf-8"))
    if draft.get("schema_version") != 2 or "draft_samples" not in draft or reviewed.get("training_ready") is not False:
        raise ValueError("Expected a review draft and frozen development video annotations")
    root = Path(draft["root"])
    if not root.is_absolute():
        root = draft_path.parent / root
    rejected = {key.removeprefix("dataset:") for key, row in session["decisions"].items()
                if key.startswith("dataset:") and row["review_status"] == "rejected"}
    candidates, masks, excluded = [], {}, Counter()
    for original in draft["draft_samples"]:
        if original["proposed_split"] != "train":
            excluded["raw_validation_or_test_groups"] += 1
            continue
        if original["sample_id"] in rejected:
            excluded["explicitly_rejected"] += 1
            continue
        image, mask_path = (root / original["image"]).resolve(), (root / original["mask"]).resolve()
        if sha256(image) != original["image_sha256"] or sha256(mask_path) != original["mask_sha256"]:
            raise ValueError("Draft content changed since audit")
        mask = read_class_mask(mask_path)
        # Provided positive polygons are provisional observations. Unreviewed
        # background is not a verified negative for either foreground class.
        partial = np.where(mask == 0, 255, mask).astype(np.uint8)
        row = {"sample_id": original["sample_id"], "image": str(image), "image_sha256": sha256(image),
               "source_dataset": original["source_dataset"], "group_id": original["group_id"],
               "source_group_verified": False, "review_status": "pending", "annotation_basis": "provided_positive_polygons_unreviewed",
               "stone_presence_review": "unknown", "video_id": None, "patient_id": None}
        candidates.append(row)
        if (partial != 255).any():
            masks[row["sample_id"]] = partial
    eligible = [row for row in candidates if row["sample_id"] in masks]
    random.Random(seed).shuffle(eligible)
    count = min(len(eligible), max(2, round(len(eligible) * labeled_fraction)))
    labeled = eligible[:count]
    selected = {row["sample_id"] for row in labeled}
    unlabeled = [row for row in candidates if row["sample_id"] not in selected]
    if not labeled or not unlabeled:
        raise ValueError("Need nonempty provisional labeled and unlabeled subsets")
    validation = []
    for original in reviewed["frames"]:
        if original["review_status"] != "approved":
            continue
        image, mask = (reviewed_path.parent / original["image"]).resolve(), (reviewed_path.parent / original["mask"]).resolve()
        if sha256(image) != original["image_sha256"] or sha256(mask) != original["mask_sha256"]:
            raise ValueError("Frozen reviewed frame changed")
        row = {"sample_id": f"reviewed_{original['video_id']}_{original['frame_index']:06d}",
               "image": str(image), "mask": str(mask), "image_sha256": sha256(image), "mask_sha256": sha256(mask),
               "video_id": original["video_id"], "patient_id": original["patient_id"],
               "frame_index": original["frame_index"], "group_id": original["patient_id"],
               "review_status": "approved", "reviewer": original["reviewer"], "source_group_verified": True,
               "stone_presence_review": original["stone_presence"], "annotation_basis": "reviewed_visible_pixels"}
        if original["video_id"] in {"video_1", "video_2"}:
            validation.append(row)
        else:
            raise ValueError("Explicit video assignment required for new sources")
    if {r["video_id"] for r in validation} != {"video_1", "video_2"} or {r["stone_presence_review"] for r in validation} != {"present", "absent"}:
        raise ValueError("Both validation-only videos and positive/negative validation frames are required")
    if {r["image_sha256"] for r in labeled + unlabeled} & {r["image_sha256"] for r in validation}:
        raise ValueError("Exact image overlap between training and validation")
    output.mkdir(parents=True)
    (output / "masks").mkdir()
    for row in labeled:
        path = output / "masks" / f"{row['sample_id']}.png"
        Image.fromarray(masks[row["sample_id"]]).save(path)
        row.update(mask=str(path), mask_sha256=sha256(path), sampling_weight=1.0 / len(labeled))
    manifest = {"schema_version": 3, "root": str(output), "classes": {"0":"background","1":"stone","2":"fiber"},
        "role": "unreviewed_development_experiment", "training_ready": False, "experiment_ready": True,
        "review_waiver": {"user_authorized": True, "instruction": "拒绝复核，按照plan进入下一步",
                          "date": "2026-10-27", "scope": "development training; no source or label review status promoted"},
        "train": {"labeled": labeled, "unlabeled": unlabeled}, "val": validation, "test": [],
        "independent_test_available": False,
        "split_policy": {"seed": seed, "raw_data": "only draft proposed train groups; grouping hints unverified",
                         "reviewed_video_1": "development validation only", "reviewed_video_2": "development validation only",
                         "user_constraint": "video_1 和 video_2 仅作为验证视频，不参与训练",
                         "raw_source_overlap_with_videos": "unverified", "patient_independent_verified": False},
        "partial_label_policy": "All unreviewed background pixels become ignore=255; retain provided positive polygons only",
        "sampling_policy": {"provided_positive_polygons": 1.0, "reviewed_positive_frames": 0.0, "reviewed_negative_frames": 0.0,
                            "method": "weighted replacement; expected proportions, not fixed per-batch quotas"},
        "provenance": {"draft_sha256": sha256(draft_path), "reviewed_snapshot_sha256": sha256(reviewed_path),
                       "session_sha256": sha256(session_path)},
        "counts": {"raw_train_pool": len(candidates), "provisional_labeled": len(labeled), "unlabeled": len(unlabeled),
                   "reviewed_train": 0, "reviewed_train_positive": 0,
                   "reviewed_train_negative": 0, "reviewed_validation": len(validation), "excluded": dict(excluded)},
        "evaluation_limitations": ["Both videos are validation only; no supervised or unlabeled training use", "Only 29 sparse development validation frames",
                                   "No reviewed background negatives in training; raw unknown background stays ignored",
                                   "Unreviewed raw labels and unverified case provenance", "No independent test claim"]}
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "summary.json", manifest["counts"])
    return output / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", default="datasets/ureter_semis_v2_review_20261027/manifest.json")
    parser.add_argument("--reviewed", default="runs/review_snapshots/video_keyframes_20261027_v2/pixel_reviews.json")
    parser.add_argument("--session", default="runs/annotation_sessions/review_20261027/session.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build(args.draft, args.reviewed, args.session, args.output))


if __name__ == "__main__":
    main()
