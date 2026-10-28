"""Run the bounded A1-SUP diagnostic and the same-grid A0 comparison."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import traceback
import zipfile

import torch

from ureter_semis.experiment import atomic_json, sha256
from ureter_semis.supervised_trainer import train_supervised
from .compare_checkpoints import compare_checkpoints
from .run_development_training import validation_annotations


def run(config_path: str, output_path: str, resume: bool = False) -> None:
    config_path, output = Path(config_path).resolve(), Path(output_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if output.exists() and not resume: raise FileExistsError(output)
    if (output / "retired_run.json").exists(): raise ValueError("Retired experiment")
    config_hash = sha256(config_path)
    if resume and json.loads((output / "job.json").read_text(encoding="utf-8"))["config_sha256"] != config_hash:
        raise ValueError("Changed launch configuration")
    args = dict(config["training"])
    manifest_path = (config_path.parent / args["manifest_path"]).resolve()
    args["manifest_path"] = str(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    annotations = validation_annotations(root / "runs/review_snapshots/video_keyframes_20261027_v2/pixel_reviews.json", manifest)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "validation_annotations.json", annotations)
    if not (output / "source_snapshot.zip").exists():
        with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for folder in ("ureter_semis", "tools", "third_party/Endo-SemiS/src/models"):
                for path in sorted((root / folder).rglob("*.py")): archive.write(path, str(path.relative_to(root)))
            archive.write(config_path, str(config_path.relative_to(root)))
    job = {"pid": os.getpid(), "experiment": config["experiment"], "config_sha256": config_hash,
           "planned_epochs": args["epochs"], "mode": "supervised_single_network", "model_count": 1,
           "unlabeled_images_used": 0, "training_video_frames": 0, "validation_frames": len(annotations["frames"])}
    def phase(name, **extra):
        job.update(phase=name, **extra); atomic_json(output / "job.json", job)
    try:
        phase("supervised_short_screen")
        checkpoint = train_supervised(**args, output_dir=str(output / "training"),
                                      resume=str(output / "training/last.pt") if resume else None)
        gc.collect(); torch.cuda.empty_cache()
        phase("same_grid_validation")
        report_path = output / "comparison.json"
        if not report_path.exists():
            baseline = json.loads((root / "runs/baselines/a0_20261027/config.json").read_text(encoding="utf-8"))
            compare_checkpoints(str(manifest_path), [baseline["checkpoint"], str(checkpoint)], str(report_path),
                                allow_unreviewed_experiment=True)
        status = json.loads((output / "training/status.json").read_text(encoding="utf-8"))
        comparison = json.loads(report_path.read_text(encoding="utf-8"))
        lines = ["# A1-SUP 短程纯监督检查", "", "仅原始图像训练；video_1/video_2 的 29 帧仅开发验证。",
                 "单 U-Net、ImageNet 初始化；无伪标签、无互学习；背景来自未审核的来源给定标签。", "",
                 "| 模型 | 结石 Dice | 光纤 Dice | 负帧误检面积比例 |", "|---|---:|---:|---:|"]
        for name, row in zip(("A0", "A1-SUP"), comparison["checkpoints"]):
            m = row["metrics"]
            lines.append(f"| {name} | {m['stone_dice']:.4f} | {m['fiber_dice']:.4f} | {m['stone_negative_fp_pixel_fraction']:.4f} |")
        lines += ["", f"完成 {status['completed_epoch']} 轮；选中第 {status['best_epoch']} 轮。",
                  f"通过全部预先冻结的开发门槛：{status['best_passed_screening']}。",
                  "指标为 PIL/256×256 模型网格口径；不构成独立病例测试。",
                  "未通过全部门槛时，不自动进入记忆阶段或延长训练。"]
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        phase("completed_short_screen", screening_passed=status["best_passed_screening"], best_checkpoint=str(checkpoint))
    except Exception as error:
        phase("failed", error=str(error)); (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output, args.resume)
