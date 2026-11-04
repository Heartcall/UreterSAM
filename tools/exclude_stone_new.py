"""Derive a kidney-only active manifest without altering historical experiments."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ureter_semis.experiment import atomic_json, sha256
from ureter_semis.source_policy import excluded_training_source, require_training_sources
from ureter_semis.supervision import audit_supervision


def build(previous_path: str | Path, output_path: str | Path) -> Path:
    previous_path, output = Path(previous_path).resolve(), Path(output_path).resolve()
    if output.exists(): raise FileExistsError(output)
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    manifest = copy.deepcopy(previous)
    removed = {}
    for split in ("labeled", "unlabeled"):
        rows = previous["train"].get(split, [])
        removed[split] = sum(excluded_training_source(r) for r in rows)
        manifest["train"][split] = [r for r in rows if not excluded_training_source(r)]
    rows = previous.get("reserved_unlabeled", [])
    removed["reserved_unlabeled"] = sum(excluded_training_source(r) for r in rows)
    manifest["reserved_unlabeled"] = [r for r in rows if not excluded_training_source(r)]
    old_root = Path(previous["root"])
    if not old_root.is_absolute(): old_root = previous_path.parent / old_root
    for row in manifest["train"]["labeled"] + manifest["train"]["unlabeled"] + manifest["reserved_unlabeled"] + manifest["val"]:
        for key in ("image", "mask", "source_mask"):
            if row.get(key): row[key] = str((old_root / row[key]).resolve())
    manifest["root"] = str(output)
    manifest["source_policy"] = {"excluded": ["stone_new", "stone_new_v4"],
        "reason": "User confirmed stone_new labels only the fiber tip; unsuitable for full visible fiber segmentation",
        "instruction": "停止使用 stone_new 数据集进行训练，原因是该数据集只标注光纤尖端",
        "scope": "supervised, unlabeled, reserved pool, pseudo-labeling, finetuning and checkpoint initialization",
        "remaining_source": "kidney_stones_fiber_v1"}
    manifest["provenance"]["before_source_exclusion_manifest_sha256"] = sha256(previous_path)
    n_labeled, n_unlabeled, n_reserved = [len(rows) for rows in (manifest["train"]["labeled"],
                                                             manifest["train"]["unlabeled"], manifest["reserved_unlabeled"])]
    manifest["counts"].update(raw_train_pool=n_labeled+n_unlabeled+n_reserved, provisional_labeled=n_labeled,
                              supervised_training=n_labeled, unlabeled=n_unlabeled, unlabeled_used=n_unlabeled,
                              reserved_unlabeled=n_reserved, removed_stone_new=removed)
    if "excluded" in manifest["counts"]:
        manifest["counts"]["historical_exclusions_before_source_policy"] = manifest["counts"].pop("excluded")
    manifest["evaluation_limitations"].append("Mixed-source historical checkpoints are not allowed as initialization; start a clean kidney-only experiment")
    require_training_sources(manifest)
    if not manifest["train"]["labeled"]: raise ValueError("No supervised images remain")
    audit = audit_supervision(output, manifest)
    output.mkdir(parents=True)
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "summary.json", manifest["counts"])
    atomic_json(output / "supervision_audit.json", audit)
    return output / "manifest.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", default="datasets/ureter_semis_a1_sup_bg_20261027/manifest.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); print(build(args.previous, args.output))
