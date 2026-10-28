"""Visualize tiny-stone, negative-frame and fiber-error validation examples."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ureter_semis.data import UreterDataset, load_manifest
from ureter_semis.experiment import atomic_json
from ureter_semis.inference import SegmentationPredictor
from ureter_semis.model import ensemble_logits


if __name__ == "__main__":
    output = ROOT / "runs/a1_sup_kidney_alt_20261028"
    root, manifest = load_manifest(ROOT / "datasets/ureter_semis_a1_sup_kidney_only_20261027/manifest.json")
    dataset = UreterDataset(root, manifest["val"], 256, "val")
    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    names = ["A0", "unet_control", "segformer_b0"]
    masks, errors = {}, {}
    for name, checkpoint in zip(names, comparison["checkpoints"]):
        predictor = SegmentationPredictor(checkpoint["checkpoint"])
        predictions, counts = [], []
        matrix = np.zeros((3, 3), dtype=np.int64)
        with torch.inference_mode():
            for batch in DataLoader(dataset, batch_size=2, shuffle=False):
                predicted = ensemble_logits(predictor.models, batch["image"].to(predictor.device)).argmax(1).cpu().numpy()
                for prediction, target in zip(predicted, batch["mask"].numpy()):
                    valid = target != 255
                    matrix += np.bincount((target[valid] * 3 + prediction[valid]).astype(int), minlength=9).reshape(3, 3)
                    counts.append({"stone_fp": int(((prediction == 1) & (target != 1) & valid).sum()),
                                   "stone_fn": int(((prediction != 1) & (target == 1)).sum()),
                                   "fiber_fp": int(((prediction == 2) & (target != 2) & valid).sum()),
                                   "fiber_fn": int(((prediction != 2) & (target == 2)).sum())})
                    predictions.append(prediction)
        assert matrix.tolist() == checkpoint["metrics"]["confusion_matrix"]
        masks[name], errors[name] = predictions, counts
        del predictor; torch.cuda.empty_cache()
    gt_counts = [int((dataset[i]["mask"] == 1).sum()) for i in range(len(dataset))]
    chosen = []
    groups = [sorted((i for i, n in enumerate(gt_counts) if 0 < n <= 128), key=lambda i: gt_counts[i]),
              sorted((i for i, n in enumerate(gt_counts) if n == 0), key=lambda i: max(errors[n][i]["stone_fp"] for n in names[1:]), reverse=True),
              sorted(range(len(dataset)), key=lambda i: max(errors[n][i]["fiber_fp"] for n in names[1:]), reverse=True)]
    for group in groups:
        for i in [i for i in group if i not in chosen][:2]: chosen.append(i)
    for i in groups[-1]:
        if len(chosen) >= 6: break
        if i not in chosen: chosen.append(i)
    canvas = Image.new("RGB", (1024, 28 + len(chosen) * 300), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 6), "Raw | Reviewed GT | U-Net final15 | SegFormer-B0 final15 (red stone / cyan fiber)", fill="black")
    for j, i in enumerate(chosen):
        row = manifest["val"][i]
        raw = np.asarray(Image.open(root / row["image"]).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR))
        target = dataset[i]["mask"].numpy()
        y = 28 + j * 300
        draw.text((4, y + 3), f"{row['sample_id']} | GT stone pixels={gt_counts[i]}", fill="black")
        for column, label in enumerate((None, target, masks["unet_control"][i], masks["segformer_b0"][i])):
            panel = raw.copy()
            if label is not None:
                for c, color in ((1, [240, 40, 40]), (2, [0, 220, 240])):
                    region = label == c
                    panel[region] = (.45 * panel[region] + .55 * np.array(color)).astype(np.uint8)
            canvas.paste(Image.fromarray(panel), (column * 256, y + 23))
            if column in (2, 3):
                e = errors[names[column - 1]][i]
                draw.text((column * 256 + 3, y + 280), f"S FP/FN {e['stone_fp']}/{e['stone_fn']} F {e['fiber_fp']}/{e['fiber_fn']}", fill="black")
    canvas.save(output / "failure_examples.png")
    atomic_json(output / "failure_examples.json", {"selection": "up to2 tiny-stone,2 negative,2 high fiber-FP frames, deduplicated; diagnostic selection only",
        "sample_ids": [manifest["val"][i]["sample_id"] for i in chosen], "grid": 256,
        "all_prediction_confusions_match_primary_comparison": True,
        "frame_errors": {name: errors[name] for name in names}})
