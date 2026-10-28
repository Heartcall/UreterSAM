"""Run S2 training, comparisons and replays after the user's review deferral."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import traceback
import zipfile

import torch

from ureter_semis.experiment import atomic_json, sha256
from ureter_semis.inference import SegmentationPredictor
from .compare_checkpoints import compare_checkpoints
from .evaluate_video import run_video
from .evaluate_video_annotations import evaluate
from .train_experiment import run as train


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validation_annotations(snapshot_path: Path, manifest: dict) -> dict:
    """Keep both user-designated videos out of supervised and unlabeled training."""
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    training = manifest["train"]["labeled"] + manifest["train"]["unlabeled"]
    if any(row.get("video_id") in {"video_1", "video_2"} or
           row.get("annotation_basis") == "reviewed_visible_pixels" for row in training):
        raise ValueError("video_1 and video_2 are validation only; video training is forbidden")
    if {row["video_id"] for row in manifest["val"]} != {"video_1", "video_2"}:
        raise ValueError("Both videos must be retained for validation")
    expected = {(row["video_id"], row["frame_index"]): row for row in manifest["val"]}
    if len(expected) != len(manifest["val"]):
        raise ValueError("Duplicate validation frame")
    training_hashes = {row["image_sha256"] for row in training}
    heldout = {**snapshot, "role": "derived_both_videos_development_validation", "frames": []}
    for row in snapshot["frames"]:
        key = (row["video_id"], row["frame_index"])
        if key not in expected:
            continue
        if row["image_sha256"] in training_hashes:
            raise ValueError("Exact validation image appears in training")
        if any(row[field] != expected[key][field] for field in ("image_sha256", "mask_sha256")):
            raise ValueError("Validation manifest differs from frozen annotations")
        absolute = dict(row)
        for field in ("image", "mask", "instance_mask"):
            if absolute.get(field):
                absolute[field] = str((snapshot_path.parent / absolute[field]).resolve())
        heldout["frames"].append(absolute)
    if len(heldout["frames"]) != len(expected):
        raise ValueError("Validation annotations are incomplete")
    return heldout


def run(config_path: str | Path, output_path: str | Path, resume: bool = False) -> None:
    config_path, output = Path(config_path).resolve(), Path(output_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (output / "retired_run.json").exists():
        raise ValueError("This run was retired after the video validation-only correction; start a clean run")
    if output.exists() and not resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    config_hash = sha256(config_path)
    if resume:
        previous = json.loads((output / "job.json").read_text(encoding="utf-8"))
        if previous["launch_config_sha256"] != config_hash:
            raise ValueError("Cannot resume a changed launch configuration")
    root = Path(__file__).resolve().parents[1]
    baseline_path = root / "runs/baselines/a0_20261027/config.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    manifest_path = (config_path.parent / config["training"]["manifest_path"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot_path = root / "runs/review_snapshots/video_keyframes_20261027_v2/pixel_reviews.json"
    heldout = validation_annotations(snapshot_path, manifest)
    heldout_path = output / "validation_annotations.json"
    if not heldout_path.exists():
        atomic_json(heldout_path, heldout)
    if not (output / "source_snapshot.zip").exists():
        with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for folder in ("ureter_semis", "tools", "third_party/Endo-SemiS/src/models"):
                for path in sorted((root / folder).rglob("*.py")):
                    archive.write(path, str(path.relative_to(root)))
            archive.write(config_path, str(config_path.relative_to(root)))
    job = {"pid": os.getpid(), "experiment": config["experiment"], "launch_config_sha256": config_hash,
           "scope": manifest["role"], "planned_epochs": config["training"]["epochs"],
           "review_deferred_by_user": True, "independent_test_available": False,
           "validation_only_videos": ["video_1", "video_2"], "video_frames_used_for_training": 0}
    def phase(name: str, **details) -> None:
        job.update(phase=name, updated_utc=datetime.now(timezone.utc).isoformat(), **details)
        atomic_json(output / "job.json", job)
        print(f"Pipeline phase: {name}", flush=True)
    try:
        training = output / "training"
        checkpoint = training / "last.pt"
        if not checkpoint.exists():
            phase("first_epoch_training")
            train(config_path, str(training), stop_after_epoch=1)
        release_memory()
        if not (output / "first_epoch_comparison.json").exists():
            phase("first_epoch_validation")
            compare_checkpoints(str(manifest_path), [baseline["checkpoint"], str(training / "best.pt")],
                                str(output / "first_epoch_comparison.json"), allow_unreviewed_experiment=True)
        release_memory()
        completed = json.loads((training / "status.json").read_text(encoding="utf-8"))["completed_epoch"]
        if completed < config["training"]["epochs"]:
            phase("training_remaining_epochs", first_epoch_checkpoint_verified=True)
            train(config_path, str(training), resume=str(checkpoint))
        release_memory()
        if not (output / "final_comparison.json").exists():
            phase("final_validation")
            compare_checkpoints(str(manifest_path), [baseline["checkpoint"], str(training / "best.pt")],
                                str(output / "final_comparison.json"), allow_unreviewed_experiment=True)
        release_memory()
        phase("video_replays")
        replay = output / "video_replay"
        replay.mkdir(exist_ok=True)
        replay_config = {**baseline, "experiment_id": config["experiment"], "checkpoint": str(training / "best.pt"),
                         "checkpoint_sha256": sha256(training / "best.pt"),
                         "inputs": [{**item, "role": "development_validation"}
                                    for item in baseline["inputs"]]}
        atomic_json(replay / "config.json", replay_config)
        predictor = SegmentationPredictor(training / "best.pt")
        for item in replay_config["inputs"]:
            destination = replay / "predictions" / item["video_id"]
            if (destination / "summary.json").exists():
                continue
            if destination.exists():
                raise FileExistsError(f"Partial replay retained; use a new output for replay recovery: {destination}")
            run_video(predictor, item, destination, replay_config, save_probabilities=False)
        del predictor
        release_memory()
        phase("native_grid_metrics")
        if not (output / "a0_native_validation.json").exists():
            evaluate(baseline_path, heldout_path, output / "a0_native_validation.json")
        if not (output / "a1_native_validation.json").exists():
            evaluate(replay / "config.json", heldout_path, output / "a1_native_validation.json")
        first = json.loads((output / "a0_native_validation.json").read_text(encoding="utf-8"))
        last = json.loads((output / "a1_native_validation.json").read_text(encoding="utf-8"))
        lines = ["# A1_DEV 单帧开发对照", "", f"video_1 和 video_2 仅作开发验证，共 {len(heldout['frames'])} 帧；均未参与训练。",
                 "从 ImageNet 初始化重新训练，未继承已纳入 video_1 的作废实验权重。",
                 "未核实训练来源，且在该验证集上选择权重；这些结果不构成独立病例测试。", "",
                 "| 模型 | 结石 Dice | 结石 Precision | 结石 Recall | 光纤 Dice |", "|---|---:|---:|---:|---:|"]
        for name, result in (("A0", first), ("A1_DEV", last)):
            stone, fiber = result["scores"]["stone"], result["scores"]["fiber"]
            values = [f"{value:.4f}" if value is not None else "未定义"
                      for value in (stone["dice"], stone["precision"], stone["recall"], fiber["dice"])]
            lines.append(f"| {name} | " + " | ".join(values) + " |")
        lines += ["", "数值在原图 400×400 网格上计算，未知像素忽略。模型网格/PIL 预处理口径另见 final_comparison.json。",
                  "", "下一步：结合召回、误检与回放检查 A1，再进入 S3；本次未自动发布或替换原有权重。"]
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        phase("completed", best_checkpoint=str(training / "best.pt"))
    except Exception as error:
        phase("failed", error=str(error))
        (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_a1_dev_videos_val_20261027.json")
    parser.add_argument("--output", default="runs/a1_dev_videos_val_20261027")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output, args.resume)


if __name__ == "__main__":
    main()
