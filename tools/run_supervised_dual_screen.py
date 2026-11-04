"""Frozen independent-member ensemble and equal-total-update single control."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback
import zipfile

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ureter_semis.data import UreterDataset, load_manifest
from ureter_semis.experiment import atomic_json, input_identity, require_training_access, sha256, source_identity
from ureter_semis.inference import SegmentationPredictor
from ureter_semis.supervised_trainer import assess, train_supervised
from ureter_semis.supervision import audit_supervision
from ureter_semis.trainer import validate
from .compare_checkpoints import compare_checkpoints
from .package_supervised_ensemble import package
from .run_development_training import validation_annotations


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def benchmark(checkpoint, root, rows, settings):
    gc.collect(); torch.cuda.empty_cache()
    predictor = SegmentationPredictor(checkpoint)
    frames = [cv2.imread(str(root / row["image"])) for row in rows]
    if any(frame is None for frame in frames):
        raise ValueError("Cannot read benchmark frame")
    for i in range(settings["warmup_frames"]):
        predictor.predict(frames[i % len(frames)])
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(settings["repeats"]):
        for frame in frames:
            start = time.perf_counter()
            prediction = predictor.predict(frame)
            samples.append({"pipeline_ms": (time.perf_counter() - start) * 1000,
                "model_ms": prediction.model_ms, "preprocess_ms": prediction.preprocess_ms,
                "postprocess_ms": prediction.postprocess_ms})
    result = {"checkpoint_sha256": sha256(Path(checkpoint)), "model_count": predictor.model_count,
              "parameter_count": sum(p.numel() for model in predictor.models for p in model.parameters()),
              "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "samples": len(samples), "warmup_frames": settings["warmup_frames"],
              "timings": {key: {"mean": float(np.mean([s[key] for s in samples])),
                                "p50": float(np.percentile([s[key] for s in samples], 50)),
                                "p95": float(np.percentile([s[key] for s in samples], 95))} for key in samples[0]},
              "raw_samples": samples, "scope": settings["input"]}
    del predictor
    gc.collect(); torch.cuda.empty_cache()
    return result


def run(config_path: str, output_path: str):
    config_path, output = Path(config_path).resolve(), Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    root = Path(__file__).resolve().parents[1]
    config = read(config_path)
    base_path = config_path.parent / config["base_config"]
    base = read(base_path)
    training = dict(base["training"])
    manifest_path = (base_path.parent / training["manifest_path"]).resolve()
    training["manifest_path"] = str(manifest_path)
    dataset_root, manifest = load_manifest(manifest_path)
    require_training_access(manifest, training["allow_unreviewed_experiment"])
    audit = audit_supervision(dataset_root, manifest)
    if audit["labeled_images"] != config["budget"]["images_per_epoch"] or manifest["train"]["unlabeled"]:
        raise ValueError("Unexpected training scope")
    annotations = validation_annotations(root / "runs/review_snapshots/video_keyframes_20261027_v2/pixel_reviews.json", manifest)
    identity, sources = input_identity(manifest_path, dataset_root, manifest), source_identity()
    # All identities and the complete budget are frozen before constructing a model.
    output.mkdir(parents=True)
    atomic_json(output / "preflight.json", {"config": config, "config_sha256": sha256(config_path),
        "base_config": base, "base_config_sha256": sha256(base_path), "inputs": identity, "source_hashes": sources,
        "planned_updates": {a["name"]: a["epochs"] * config["budget"]["updates_per_epoch"] for a in config["arms"]},
        "frozen_before_model_construction": True})
    atomic_json(output / "validation_annotations.json", annotations)
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("ureter_semis", "tools", "third_party/Endo-SemiS/src/models"):
            for path in sorted((root / folder).rglob("*.py")):
                archive.write(path, str(path.relative_to(root)))
        for path in (config_path, base_path): archive.write(path, str(path.relative_to(root)))
    job = {"pid": os.getpid(), "phase": "prepared", "completed_arms": [],
           "experiment": config["experiment"], "training_video_frames": 0, "unlabeled_images_used": 0,
           "validation_frames": len(manifest["val"])}
    def phase(name, **extra):
        job.update(phase=name, **extra); atomic_json(output / "job.json", job)
    try:
        timings = {}
        for arm in config["arms"]:
            phase("training", current_arm=arm["name"])
            args = dict(training, seed=arm["seed"], epochs=arm["epochs"])
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            train_supervised(**args, output_dir=str(output / arm["name"]))
            elapsed = time.perf_counter() - started
            history = read(output / arm["name"] / "history.json")
            timings[arm["name"]] = {"train_validation_checkpoint_wall_s": elapsed,
                "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "optimizer_updates": sum(row["optimizer_update_steps"] for row in history),
                "training_image_exposures": sum(row["sampled_training_images"] for row in history)}
            atomic_json(output / "training_cost.json", timings)
            if len(history) != arm["epochs"]:
                raise RuntimeError("Arm stopped early: equal-budget comparison unavailable")
            job["completed_arms"].append(arm["name"])
            phase("arm_completed", current_arm=arm["name"])
        phase("packaging", current_arm=None)
        ensemble_path = package(str(manifest_path),
            [str(output / name / "last.pt") for name in config["ensemble_members"]], str(output / "dual_final.pt"))
        checkpoints = {"A0": Path(base["baseline_checkpoint"]),
                       **{a["name"]: output / a["name"] / "last.pt" for a in config["arms"]},
                       "dual15": ensemble_path}
        phase("same_grid_validation")
        comparison = compare_checkpoints(str(manifest_path), [str(p) for p in checkpoints.values()],
                                          str(output / "comparison.json"), allow_unreviewed_experiment=True)
        by_video = {}
        for name, path in checkpoints.items():
            predictor = SegmentationPredictor(path)
            by_video[name] = {video: validate(predictor.models, DataLoader(
                UreterDataset(dataset_root, [row for row in manifest["val"] if row["video_id"] == video], 256, "val"),
                batch_size=2, shuffle=False), predictor.device) for video in ("video_1", "video_2")}
            del predictor
            gc.collect(); torch.cuda.empty_cache()
        atomic_json(output / "per_video.json", by_video)
        phase("benchmark")
        performance = {}
        for name, path in checkpoints.items():
            performance[name] = benchmark(path, dataset_root, manifest["val"], config["performance"])
            atomic_json(output / "performance.json", performance)
        # A new source edit or input mutation invalidates the matched experiment.
        if source_identity() != sources or input_identity(manifest_path, dataset_root, manifest) != identity:
            raise RuntimeError("Source or data changed during the experiment")
        rows = []
        for (name, path), row in zip(checkpoints.items(), comparison["checkpoints"]):
            rows.append({"name": name, "checkpoint": str(path), "metrics": row["metrics"],
                         "screening": assess(row["metrics"], training["gates"]),
                         "performance": {k: v for k, v in performance[name].items() if k != "raw_samples"}})
        atomic_json(output / "summary.json", {"primary_checkpoints": config["primary_checkpoints"],
            "budget": config["budget"], "models": rows, "training_cost": timings,
            "scope": "Repeated development validation, provisional training labels, no independent-case generalization claim"})
        lines = ["# A1-DUAL 固定预算开发对照", "", "190 张 kidney 监督图像；29 帧仅验证；末轮固定，未搜索成员组合。", "",
                 "| 模型 | 结石 Dice | 光纤 Dice | 负帧误检面积 | 推理 p50 / p95 ms | 全部门槛 |",
                 "|---|---:|---:|---:|---:|---|"]
        for row in rows:
            m, p = row["metrics"], row["performance"]["timings"]["pipeline_ms"]
            lines.append(f"| {row['name']} | {m['stone_dice']:.4f} | {m['fiber_dice']:.4f} | "
                         f"{m['stone_negative_fp_pixel_fraction']:.4%} | {p['p50']:.2f} / {p['p95']:.2f} | {row['screening']['passed']} |")
        lines += ["", "推理成本是本机预加载帧到概率/掩膜，不含视频解码、显示、几何及记忆。",
                  "双网总更新数 2850，与新单网30轮一致；双网与单网15轮相比使用双倍更新数。",
                  "训练耗时包含初始化、逐轮验证和断点写入；两成员顺序训练。未通过门槛不进入 S3。"]
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        phase("completed", screening_passed=next(r["screening"]["passed"] for r in rows if r["name"] == "dual15"))
    except Exception as error:
        phase("failed", error=str(error))
        (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); run(args.config, args.output)
