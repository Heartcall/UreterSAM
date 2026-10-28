"""Localize fiber errors on fixed PIL grids; never update weights or labels."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from ureter_semis.data import UreterDataset, load_manifest
from ureter_semis.experiment import atomic_json, input_identity, require_training_access, seed_training, sha256
from ureter_semis.inference import SegmentationPredictor
from ureter_semis.model import ensemble_logits


def fiber_errors(target: np.ndarray, prediction: np.ndarray, radius: float = 3.) -> dict:
    if target.shape != prediction.shape or target.ndim != 2 or radius < 0:
        raise ValueError("Need equal 2D masks and nonnegative radius")
    valid, truth = target != 255, target == 2
    predicted = prediction == 2
    fp = valid & predicted & ~truth
    distance = (cv2.distanceTransform((~truth).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                if truth.any() else np.full(truth.shape, np.inf))
    return {"tp": int((valid & predicted & truth).sum()), "fp": int(fp.sum()),
            "fn": int((valid & ~predicted & truth).sum()),
            "fp_over_background": int((fp & (target == 0)).sum()),
            "fp_over_stone": int((fp & (target == 1)).sum()),
            "fp_within_3px_of_fiber": int((fp & (distance <= radius)).sum()),
            "fp_farther_than_3px": int((fp & (distance > radius)).sum())}


def summarize(records: list[dict]) -> dict:
    counts = {k: sum(r[k] for r in records) for k in
              ("tp", "fp", "fn", "fp_over_background", "fp_over_stone", "fp_within_3px_of_fiber", "fp_farther_than_3px")}
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    counts.update(images=len(records), images_with_fp=sum(r["fp"] > 0 for r in records),
                  fiber_dice=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                  fiber_precision=tp / (tp + fp) if tp + fp else None,
                  fiber_recall=tp / (tp + fn) if tp + fn else None,
                  near_fp_fraction=counts["fp_within_3px_of_fiber"] / fp if fp else None)
    return counts


def contact_sheet(root: Path, rows: list[dict], dataset, records: list[dict], mask_folder: Path, output: Path):
    worst = sorted(records, key=lambda r: r["fp"], reverse=True)[:6]
    canvas = Image.new("RGB", (1024, len(worst) * 290 + 28), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 6), "Raw | Reference fiber (cyan) | Prediction fiber (cyan) | FP magenta / FN yellow", fill="black")
    for j, record in enumerate(worst):
        i = record["index"]
        raw = np.asarray(Image.open(root / rows[i]["image"]).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)).copy()
        truth = dataset[i]["mask"].numpy()
        prediction = np.asarray(Image.open(mask_folder / f"{i:04d}.png"))
        valid = truth != 255
        panels = [raw.copy() for _ in range(4)]
        for panel, region in ((panels[1], truth == 2), (panels[2], prediction == 2)):
            panel[region] = (.4 * panel[region] + .6 * np.array([0, 220, 240])).astype(np.uint8)
        panels[3][valid & (prediction == 2) & (truth != 2)] = [245, 0, 210]
        panels[3][valid & (prediction != 2) & (truth == 2)] = [255, 225, 0]
        y = 28 + j * 290
        draw.text((5, y + 3), f"{record['sample_id'][:85]} | FP={record['fp']} FN={record['fn']}", fill="black")
        for column, panel in enumerate(panels):
            canvas.paste(Image.fromarray(panel), (column * 256, y + 27))
    canvas.save(output)


def diagnose(manifest_path: str, checkpoints: list[str], output_path: str):
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    root, manifest = load_manifest(manifest_path)
    require_training_access(manifest, True)
    parsed = [spec.split("=", 1) for spec in checkpoints]
    if any(len(p) != 2 or not p[0].replace("_", "").isalnum() for p in parsed) or len({p[0] for p in parsed}) != len(parsed):
        raise ValueError("Use distinct simple names: --checkpoint name=path")
    output.mkdir(parents=True)
    seed_training(42)
    result = {"grid": 256, "preprocessing": "PIL bilinear RGB / nearest labels; same as training validation",
              "near_fp_definition": "FP within Euclidean 3 pixels of reference fiber on the 256 grid; diagnostic, not clinical tolerance",
              "training_reference": "Provisional source labels; disagreements do not confirm annotation errors",
              "scope": "Read-only diagnosis; no weight, label or split updates; reserved pool not read",
              "inputs": input_identity(Path(manifest_path).resolve(), root, manifest), "models": {}}
    for name, checkpoint in parsed:
        predictor = SegmentationPredictor(checkpoint)
        if predictor.size != 256:
            raise ValueError("This fixed diagnostic requires 256 grids")
        model_result = {"checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha256(Path(checkpoint)), "splits": {}}
        for split, rows in (("train", manifest["train"]["labeled"]), ("val", manifest["val"])):
            dataset = UreterDataset(root, rows, 256, "val")
            loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
            mask_folder = output / name / split
            mask_folder.mkdir(parents=True)
            records = []
            with torch.inference_mode():
                for batch in loader:
                    masks = ensemble_logits(predictor.models, batch["image"].to(predictor.device)).argmax(1).byte().cpu().numpy()
                    for target, prediction in zip(batch["mask"].numpy(), masks):
                        index = len(records)
                        row = rows[index]
                        record = {"index": index, "sample_id": row.get("sample_id", f"{row.get('video_id')}/{row.get('frame_index')}"),
                                  "video_id": row.get("video_id"), "frame_index": row.get("frame_index"),
                                  **fiber_errors(target, prediction)}
                        records.append(record)
                        Image.fromarray(prediction).save(mask_folder / f"{index:04d}.png")
            groups = defaultdict(list)
            for record in records:
                groups[record["video_id"] or "kidney_source"].append(record)
            model_result["splits"][split] = {"overall": summarize(records),
                "groups": {group: summarize(items) for group, items in groups.items()},
                "frames": sorted(records, key=lambda r: r["fp"], reverse=True)}
            if name != "a0":
                contact_sheet(root, rows, dataset, records, mask_folder, output / f"{name}_{split}_worst_fp.png")
        result["models"][name] = model_result
        atomic_json(output / "diagnosis.json", result)
        print(name, {split: v["overall"] for split, v in model_result["splits"].items()}, flush=True)
        del predictor
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    diagnose(args.manifest, args.checkpoint, args.output)
