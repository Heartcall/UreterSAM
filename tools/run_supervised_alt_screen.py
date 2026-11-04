"""Freeze, train and compare a same-data full-architecture alternative."""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
from pathlib import Path
import time
import traceback
import zipfile

import segmentation_models_pytorch as smp
import torch

from ureter_semis.data import load_manifest
from ureter_semis.experiment import atomic_json, input_identity, require_training_access, sha256, source_identity
from ureter_semis.supervised_trainer import assess, train_supervised
from ureter_semis.supervision import audit_supervision
from .compare_checkpoints import compare_checkpoints
from .evaluate_spatial_detail import evaluate as evaluate_detail
from .run_development_training import validation_annotations
from .run_supervised_dual_screen import benchmark


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
        raise ValueError("Frozen supervised-only scope changed")
    annotations = validation_annotations(root / "runs/review_snapshots/video_keyframes_20261027_v2/pixel_reviews.json", manifest)
    encoder_path = (config_path.parent / config["arms"][1]["encoder_checkpoint"]).resolve()
    if sha256(encoder_path) != config["encoder_sha256"]: raise ValueError("Changed MiT encoder")
    resnet_path = Path(torch.hub.get_dir()) / "checkpoints/resnet34-333f7ec4.pth"
    initialization_files = {str(p): sha256(p) for p in (encoder_path, resnet_path)}
    identity, sources = input_identity(manifest_path, dataset_root, manifest), source_identity()
    from segmentation_models_pytorch.encoders import mix_transformer
    from segmentation_models_pytorch.decoders.segformer import decoder
    implementation = {inspect.getfile(module): sha256(Path(inspect.getfile(module))) for module in (smp.Segformer, mix_transformer, decoder)}
    output.mkdir(parents=True)
    atomic_json(output / "preflight.json", {"config": config, "config_sha256": sha256(config_path),
        "base_config": base, "base_config_sha256": sha256(base_path), "inputs": identity, "source_hashes": sources,
        "initialization_files": initialization_files, "installed_implementation": implementation,
        "planned_updates_per_arm": 1425, "frozen_before_model_construction": True})
    atomic_json(output / "validation_annotations.json", annotations)
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("ureter_semis", "tools", "third_party/Endo-SemiS/src/models"):
            for path in sorted((root / folder).rglob("*.py")): archive.write(path, str(path.relative_to(root)))
        for path in (config_path, base_path): archive.write(path, str(path.relative_to(root)))
    job = {"pid": os.getpid(), "experiment": config["experiment"], "completed_arms": [],
           "training_video_frames": 0, "unlabeled_images_used": 0, "validation_frames": 29}
    def phase(name, **extra):
        job.update(phase=name, **extra); atomic_json(output / "job.json", job)
    try:
        cost = {}
        for arm in config["arms"]:
            phase("training", current_arm=arm["name"])
            training = dict(args, architecture=arm["architecture"], epochs=arm["epochs"], seed=arm["seed"])
            if "encoder_checkpoint" in arm: training["encoder_checkpoint"] = str(encoder_path)
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            train_supervised(**training, output_dir=str(output / arm["name"]))
            history = read(output / arm["name"] / "history.json")
            cost[arm["name"]] = {"training_validation_io_wall_s": time.perf_counter() - started,
                "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "optimizer_updates": sum(r["optimizer_update_steps"] for r in history),
                "image_exposures": sum(r["sampled_training_images"] for r in history)}
            atomic_json(output / "training_cost.json", cost)
            if len(history) != arm["epochs"]: raise RuntimeError("Stopped early; equal-budget comparison unavailable")
            job["completed_arms"].append(arm["name"])
        phase("same_grid_validation", current_arm=None)
        checkpoints = {"A0": Path(base["baseline_checkpoint"]),
                       **{a["name"]: output / a["name"] / "last.pt" for a in config["arms"]}}
        comparison = compare_checkpoints(str(manifest_path), [str(p) for p in checkpoints.values()],
            str(output / "comparison.json"), allow_unreviewed_experiment=True)
        phase("spatial_detail")
        detail = evaluate_detail(str(manifest_path), checkpoints, config["spatial_detail"]["boundary_tolerance_pixels"],
                                 config["spatial_detail"]["small_stone_max_gt_pixels"])
        atomic_json(output / "spatial_detail.json", detail)
        phase("benchmark")
        performance = {}
        for name, path in checkpoints.items():
            performance[name] = benchmark(path, dataset_root, manifest["val"], config["performance"])
            atomic_json(output / "performance.json", performance)
        if (source_identity() != sources or input_identity(manifest_path, dataset_root, manifest) != identity or
                any(sha256(Path(p)) != digest for p, digest in initialization_files.items())):
            raise RuntimeError("Inputs, initialization or source mutated")
        models = []
        for (name, path), row in zip(checkpoints.items(), comparison["checkpoints"]):
            models.append({"name": name, "checkpoint": str(path), "metrics": row["metrics"],
                "screening": assess(row["metrics"], args["gates"]),
                "spatial_detail": {k: v for k, v in detail["models"][name].items() if k != "frames"},
                "performance": {k: v for k, v in performance[name].items() if k != "raw_samples"}})
        atomic_json(output / "summary.json", {"primary_comparison": config["primary_comparison"],
            "models": models, "training_cost": cost, "scope": "Development short screen, not model upper bound or independent generalization"})
        lines = ["# A1-ALT 同源监督对照", "", "固定末轮15轮、190张监督、1425次更新；只换完整网络，未调整损失/分辨率。", "",
            "| 模型 | 结石Dice | 光纤Dice | 负帧误检面积 | 推理p50/p95 ms | 全部门槛 |", "|---|---:|---:|---:|---:|---|"]
        for row in models:
            m, p = row["metrics"], row["performance"]["timings"]["pipeline_ms"]
            lines.append(f"| {row['name']} | {m['stone_dice']:.4f} | {m['fiber_dice']:.4f} | "
                         f"{m['stone_negative_fp_pixel_fraction']:.4%} | {p['p50']:.2f}/{p['p95']:.2f} | {row['screening']['passed']} |")
        lines += ["", "使用SMP0.4 SegFormer-MiT-B0实现，非官方训练配方复现或纯编码器消融。",
                  "推理成本不含解码、显示、几何、记忆。未通过共同门槛不进入S3，不自动延长。"]
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        phase("completed", screening_passed=models[-1]["screening"]["passed"])
    except Exception as error:
        phase("failed", error=str(error)); (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8"); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); run(args.config, args.output)
