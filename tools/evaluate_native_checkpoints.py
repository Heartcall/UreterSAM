"""Compare different input resolutions on the same untouched native masks."""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from ureter_semis.data import load_manifest
from ureter_semis.experiment import atomic_json, input_identity, sha256, seed_training
from ureter_semis.inference import SegmentationPredictor
from ureter_semis.native_validation import NativeValidationDataset, collate_native, validate_native
from .evaluate_spatial_detail import boundary_counts, boundary_summary


def evaluate(manifest_path, checkpoints, output_path, boundary_tolerance=3, area_bins=(0, 1024, 4096)):
    output = Path(output_path).resolve()
    if output.exists(): raise FileExistsError(output)
    root, manifest = load_manifest(manifest_path)
    output.mkdir(parents=True)
    seed_training(42)
    result = {"protocol": "PIL input resize / mean logits / softmax / FP32 bilinear probabilities to native, align_corners=false / argmax; GT unchanged",
        "inputs": input_identity(Path(manifest_path).resolve(), root, {"train": {"labeled": [], "unlabeled": []}, "val": manifest["val"]}),
        "boundary_tolerance_native_pixels": boundary_tolerance, "area_bins_native_pixels": list(area_bins), "models": {}}
    for name, checkpoint in checkpoints.items():
        predictor = SegmentationPredictor(checkpoint)
        dataset = NativeValidationDataset(root, manifest["val"], predictor.size)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_native)
        metrics, frames = validate_native(predictor.models, loader, predictor.device, collect_frames=True)
        mask_folder = output / name; mask_folder.mkdir()
        details = []
        for index, frame in enumerate(frames):
            prediction, target = frame["prediction"], frame["target"]
            Image.fromarray(prediction).save(mask_folder / f"{index:04d}.png")
            truth = target == 1; valid = target != 255
            details.append({"sample_id": frame["sample_id"], "native_shape": list(target.shape),
                "gt_stone_pixels": int(truth.sum()), "stone_tp": int(((prediction == 1) & truth).sum()),
                "stone_fp": int(((prediction == 1) & ~truth & valid).sum()),
                "fiber_fp": int(((prediction == 2) & (target != 2) & valid).sum()),
                "fiber_fn": int(((prediction != 2) & (target == 2)).sum()),
                "boundaries": {str(c): boundary_counts(prediction, target, c, boundary_tolerance) for c in (1, 2)}})
        groups = {}
        for lower, upper in zip(area_bins, (*area_bins[1:], None)):
            selected = [r for r in details if r["gt_stone_pixels"] > lower and (upper is None or r["gt_stone_pixels"] <= upper)]
            total, tp = sum(r["gt_stone_pixels"] for r in selected), sum(r["stone_tp"] for r in selected)
            groups[f"({lower},{upper if upper is not None else 'inf'}]"] = {"frames": len(selected), "target_pixels": total,
                "tp": tp, "pixel_recall": tp / total if total else None, "frames_any_tp": sum(r["stone_tp"] > 0 for r in selected)}
        result["models"][name] = {"checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha256(Path(checkpoint)),
            "input_size": predictor.size, "metrics": metrics,
            "boundaries": {str(c): boundary_summary([r["boundaries"][str(c)] for r in details]) for c in (1, 2)},
            "visible_stone_area_groups": groups, "frames": details}
        atomic_json(output / "comparison.json", result)
        del predictor
        gc.collect(); torch.cuda.empty_cache()
    return result
