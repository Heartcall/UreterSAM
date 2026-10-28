"""Run frozen, equal-budget fiber ablations without changing the dataset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ureter_semis.experiment import atomic_json, sha256
from .run_supervised_screen import run


def execute(suite_path: str, output_path: str) -> None:
    suite_path, output = Path(suite_path).resolve(), Path(output_path).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    job = {"pid": os.getpid(), "suite_sha256": sha256(suite_path), "phase": "training_ablations",
           "planned_arms": [row["name"] for row in suite["arms"]], "completed_arms": []}
    atomic_json(output / "suite.json", suite)
    atomic_json(output / "job.json", job)
    try:
        summaries = []
        for arm in suite["arms"]:
            config = (suite_path.parent / arm["config"]).resolve()
            job["current_arm"] = arm["name"]; atomic_json(output / "job.json", job)
            run(str(config), str(output / arm["name"]))
            history = json.loads((output / arm["name"] / "training/history.json").read_text(encoding="utf-8"))
            status = json.loads((output / arm["name"] / "training/status.json").read_text(encoding="utf-8"))
            last = history[-1]
            summaries.append({"arm": arm["name"], "factor": arm["factor"], "completed_epoch": last["epoch"],
                "last_epoch_metrics": last["metrics"], "selected_epoch": status["best_epoch"],
                "selected_epoch_metrics": history[status["best_epoch"] - 1]["metrics"],
                "maximum_fiber_dice_diagnostic_only": max(r["metrics"]["fiber_dice"] or 0 for r in history),
                "screening_passed": status["best_passed_screening"],
                "sampled_fiber_image_fraction": sum(r["sampled_fiber_images"] for r in history) /
                    sum(r["sampled_training_images"] for r in history)})
            job["completed_arms"].append(arm["name"])
            atomic_json(output / "job.json", job)
            atomic_json(output / "summary.json", {"scope": "same split, seed and budget; provisional labels; development only",
                "primary_comparison": "same final epoch; selected checkpoints separately reported", "arms": summaries})
        lines = ["# 光纤学习单因素对照", "", "同一 350 张训练图像、29 帧仅验证、seed 42；每组从 ImageNet 初始化、相同 15 轮计划。",
                 "主要比较相同最终轮次，最佳轮次和诊断最大光纤分数另存 summary.json。", "",
                 "| 组别 | 完成轮数 | 结石 Dice | 光纤 Dice | 负帧误检面积 | 训练抽样中有光纤图像比例 | 全部门槛通过 |",
                 "|---|---:|---:|---:|---:|---:|---|"]
        for row in summaries:
            m = row["last_epoch_metrics"]
            lines.append(f"| {row['arm']} | {row['completed_epoch']} | {m['stone_dice']:.4f} | {m['fiber_dice']:.4f} | "
                         f"{m['stone_negative_fp_pixel_fraction']:.4%} | {row['sampled_fiber_image_fraction']:.1%} | {row['screening_passed']} |")
        lines += ["", "训练图像及标签不变，来源标签仍待审；不构成独立病例测试。未通过共同门槛不进入 S3。"]
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        job.update(phase="completed", current_arm=None); atomic_json(output / "job.json", job)
    except Exception as error:
        job.update(phase="failed", error=str(error)); atomic_json(output / "job.json", job)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); execute(args.suite, args.output)
