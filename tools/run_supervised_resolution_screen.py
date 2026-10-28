"""Fixed-update U-Net256/512 screen with native GT and freshly frozen A0 gates."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

import torch

from ureter_semis.data import load_manifest
from ureter_semis.experiment import atomic_json, input_identity, require_training_access, sha256, source_identity
from ureter_semis.supervised_trainer import assess, train_supervised
from ureter_semis.supervision import audit_supervision
from .compare_checkpoints import compare_checkpoints
from .evaluate_native_checkpoints import evaluate
from .run_development_training import validation_annotations


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def run(config_path, output_path):
    config_path, output = Path(config_path).resolve(), Path(output_path).resolve()
    if output.exists(): raise FileExistsError(output)
    root = Path(__file__).resolve().parents[1]
    config = read(config_path); base_path = config_path.parent / config["base_config"]; base = read(base_path)
    args = dict(base["training"]); manifest_path = (base_path.parent / args["manifest_path"]).resolve()
    args["manifest_path"] = str(manifest_path)
    dataset_root, manifest = load_manifest(manifest_path)
    require_training_access(manifest, True); audit_supervision(dataset_root, manifest)
    if len(manifest["train"]["labeled"]) != 190 or manifest["train"]["unlabeled"]:
        raise ValueError("Frozen supervised scope changed")
    annotations = validation_annotations(root / "runs/review_snapshots/video_keyframes_20261027_v2/pixel_reviews.json", manifest)
    identity, sources = input_identity(manifest_path, dataset_root, manifest), source_identity()
    encoder = Path(torch.hub.get_dir()) / "checkpoints/resnet34-333f7ec4.pth"
    encoder_hash = sha256(encoder)
    output.mkdir(parents=True)
    preflight = {"config": config, "config_sha256": sha256(config_path), "base_config": base,
        "base_config_sha256": sha256(base_path), "inputs": identity, "source_hashes": sources,
        "pretrained_encoder": {str(encoder): encoder_hash}, "planned_updates_per_arm": 1425,
        "protocol_frozen_before_a0_evaluation": True}
    atomic_json(output / "preflight.json", preflight); atomic_json(output / "validation_annotations.json", annotations)
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("ureter_semis", "tools", "third_party/Endo-SemiS/src/models"):
            for path in sorted((root / folder).rglob("*.py")): archive.write(path, str(path.relative_to(root)))
        for path in (config_path, base_path): archive.write(path, str(path.relative_to(root)))
    job = {"pid": os.getpid(), "experiment": config["experiment"], "completed_arms": [],
        "training_video_frames": 0, "unlabeled_images_used": 0, "validation_frames": 29}
    def phase(name, **extra):
        job.update(phase=name, **extra); atomic_json(output / "job.json", job)
    try:
        phase("freeze_native_a0_reference")
        detail = config["native_protocol"]
        baseline = evaluate(str(manifest_path), {"A0": Path(base["baseline_checkpoint"])}, str(output / "baseline_native"),
            detail["boundary_tolerance_native_pixels"], tuple(detail["small_visible_stone_area_bins_native_pixels"]))
        m = baseline["models"]["A0"]["metrics"]
        gates = {"stone_dice_min": m["stone_dice"], "fiber_dice_min": m["fiber_dice"] - .01,
                 "negative_fp_fraction_max": m["stone_negative_fp_pixel_fraction"]}
        preflight.update(native_gates=gates, native_a0_metrics=m, gates_frozen_before_training_model_construction=True)
        atomic_json(output / "preflight.json", preflight)
        costs = {}
        for arm in config["arms"]:
            phase("training", current_arm=arm["name"])
            training = dict(args, size=arm["size"], epochs=arm["epochs"], seed=arm["seed"],
                            validation_grid="native_probability", gates=gates)
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            train_supervised(**training, output_dir=str(output / arm["name"]))
            history = read(output / arm["name"] / "history.json")
            costs[arm["name"]] = {"training_validation_io_wall_s": time.perf_counter()-started,
                "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
                "optimizer_updates": sum(r["optimizer_update_steps"] for r in history),
                "image_exposures": sum(r["sampled_training_images"] for r in history)}
            atomic_json(output / "training_cost.json", costs)
            if len(history) != arm["epochs"]: raise RuntimeError("Arm stopped early; equal-budget comparison unavailable")
            job["completed_arms"].append(arm["name"])
        phase("native_comparison", current_arm=None)
        checkpoints = {"A0": Path(base["baseline_checkpoint"]),
                       **{a["name"]: output / a["name"] / "last.pt" for a in config["arms"]}}
        native = evaluate(str(manifest_path), checkpoints, str(output / "native_comparison"),
            detail["boundary_tolerance_native_pixels"], tuple(detail["small_visible_stone_area_bins_native_pixels"]))
        phase("legacy256_regression")
        compare_checkpoints(str(manifest_path), [str(checkpoints["A0"]), str(checkpoints["unet256"])],
            str(output / "legacy256_comparison.json"), allow_unreviewed_experiment=True)
        phase("fresh_process_native_benchmark")
        perf = {}
        for name, checkpoint in checkpoints.items():
            path = output / f"performance_{name}.json"
            subprocess.run([sys.executable, "-m", "tools.benchmark_native", "--manifest", str(manifest_path),
                "--checkpoint", str(checkpoint), "--output", str(path), "--warmup", str(config["performance"]["warmup_frames"]),
                "--repeats", str(config["performance"]["repeats"])], cwd=root, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            perf[name] = read(path); atomic_json(output / "performance.json", perf)
        if source_identity() != sources or input_identity(manifest_path, dataset_root, manifest) != identity or sha256(encoder) != encoder_hash:
            raise RuntimeError("Source, data or pretrained encoder changed")
        models = [{"name": name, **{k: v for k, v in row.items() if k != "frames"},
                   "screening": assess(row["metrics"], gates),
                   "performance": {k: v for k, v in perf[name].items() if k != "raw_samples"}}
                  for name, row in native["models"].items()]
        atomic_json(output / "summary.json", {"protocol": config["native_protocol"], "native_gates": gates,
            "models": models, "training_cost": costs, "scope": "Native400 development validation; not independent cases, instance truth or physical-size measurement"})
        lines = ["# U-Net256/512 原图网格对照", "", "原图GT不缩放；概率双线性映射回原图后argmax。与旧256网格指标分开。", "",
                 "| 模型 | 结石Dice | 光纤Dice | 负帧误检面积 | 原图输出p50/p95 ms | 全部门槛 |", "|---|---:|---:|---:|---:|---|"]
        for row in models:
            m, p = row["metrics"], row["performance"]["timings"]["pipeline_ms"]
            lines.append(f"| {row['name']} | {m['stone_dice']:.4f} | {m['fiber_dice']:.4f} | {m['stone_negative_fp_pixel_fraction']:.4%} | {p['p50']:.2f}/{p['p95']:.2f} | {row['screening']['passed']} |")
        lines += ["", "两组各15轮1425次更新；512输入有4倍像素，非等硬件成本。800张备用图像和两段视频均未训练。",
                  "指标仅29帧开发验证；通过筛选也需多种子及失败片段验证。未通过则不自动延长或进入S3。"]
        (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        phase("completed", screening_passed=models[-1]["screening"]["passed"])
    except Exception as error:
        phase("failed", error=str(error)); (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8"); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); run(args.config, args.output)
