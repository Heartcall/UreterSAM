"""Restore source-provided provisional background on the frozen labeled subset."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ureter_semis.experiment import atomic_json, sha256
from ureter_semis.labels import read_class_mask
from ureter_semis.supervision import audit_supervision


def build(previous_path: str | Path, draft_path: str | Path, session_path: str | Path,
          output_path: str | Path) -> Path:
    previous_path, draft_path, session_path, output = [Path(p).resolve() for p in
                                                     (previous_path, draft_path, session_path, output_path)]
    if output.exists():
        raise FileExistsError(output)
    previous, draft, session = [json.loads(p.read_text(encoding="utf-8")) for p in
                               (previous_path, draft_path, session_path)]
    root = Path(draft["root"])
    if not root.is_absolute():
        root = draft_path.parent / root
    originals = {r["sample_id"]: r for r in draft["draft_samples"]}
    rejected = {k.removeprefix("dataset:") for k, r in session["decisions"].items()
                if k.startswith("dataset:") and r["review_status"] == "rejected"}
    pool = previous["train"]["labeled"] + previous["train"]["unlabeled"]
    if any(r["sample_id"] in rejected or originals[r["sample_id"]]["proposed_split"] != "train" for r in pool):
        raise ValueError("The frozen training pool includes a rejected or non-training source")
    if any(r.get("video_id") is not None for r in pool):
        raise ValueError("Frozen pool must contain raw images only")
    manifest = copy.deepcopy(previous)
    manifest.update(root=str(output), training_mode="supervised_single_network",
                    reserved_unlabeled=copy.deepcopy(previous["train"]["unlabeled"]))
    manifest["train"] = {"labeled": [], "unlabeled": []}
    manifest["partial_label_policy"] = (
        "Retain source-provided polygons and provisional background=0; preserve existing unknown=255. "
        "Background is not reviewed truth; no validation video is used for training.")
    manifest["sampling_policy"] = {"method": "uniform shuffle without replacement", "video_training_mass": 0}
    manifest["review_waiver"]["instruction"] = "同意修改 plan；按照 plan 开始执行下一步"
    manifest["provenance"].update(previous_manifest_sha256=sha256(previous_path), draft_sha256=sha256(draft_path),
                                  session_sha256=sha256(session_path))
    prepared = []
    for old in previous["train"]["labeled"]:
        source = originals[old["sample_id"]]
        image, mask_path = (root / source["image"]).resolve(), (root / source["mask"]).resolve()
        if sha256(image) != source["image_sha256"] or sha256(mask_path) != source["mask_sha256"]:
            raise ValueError("Source image/mask changed since draft freeze")
        mask = read_class_mask(mask_path)
        if not np.any((mask == 1) | (mask == 2)):
            raise ValueError("Do not promote an unreviewed empty image to a verified negative")
        row = {**copy.deepcopy(old), "image": str(image), "source_mask": str(mask_path),
               "source_mask_sha256": source["mask_sha256"], "review_status": "pending",
               "source_group_verified": False, "stone_presence_review": "unknown",
               "annotation_basis": "provided_polygons_and_provisional_background_unreviewed",
               "background_basis": "source_label_background_provisional_not_reviewed",
               "review_flags": source.get("review_flags", []),
               "fiber_annotation_completeness": source.get("fiber_annotation_completeness", "unknown")}
        row.pop("sampling_weight", None)
        prepared.append((row, mask))
    output.mkdir(parents=True)
    (output / "masks").mkdir()
    for row, mask in prepared:
        path = output / "masks" / f"{row['sample_id']}.png"
        Image.fromarray(mask).save(path)
        row.update(mask=str(path), mask_sha256=sha256(path))
        manifest["train"]["labeled"].append(row)
    audit = audit_supervision(output, manifest)
    train_hashes = {r["image_sha256"] for r in pool}
    if train_hashes & {r["image_sha256"] for r in manifest["val"]}:
        raise ValueError("Validation image appears in the current or reserved training pool")
    manifest["counts"].update(supervised_training=len(prepared), unlabeled_used=0,
                              reserved_unlabeled=len(manifest["reserved_unlabeled"]),
                              unlabeled=0, provisional_labeled=len(prepared), reviewed_train=0)
    manifest["evaluation_limitations"] = [
        "Both videos are validation only, including all derivatives and pseudo-labels",
        "Source-provided background is provisional and missing fiber labels may remain",
        "Raw source/case overlap is unverified; no independent test claim",
        "Reserved unlabeled pool is not read or used by the supervised experiment"]
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "supervision_audit.json", audit)
    atomic_json(output / "summary.json", manifest["counts"])
    return output / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", default="datasets/ureter_semis_a1_dev_videos_val_20261027/manifest.json")
    parser.add_argument("--draft", default="datasets/ureter_semis_v2_review_20261027/manifest.json")
    parser.add_argument("--session", default="runs/annotation_sessions/review_20261027/session.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build(args.previous, args.draft, args.session, args.output))


if __name__ == "__main__":
    main()
