"""Native-coordinate failure panels; display scaling is separate from evaluation."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ureter_semis.labels import read_class_mask
from ureter_semis.experiment import atomic_json


if __name__ == "__main__":
    output = ROOT / "runs/a1_sup_kidney_resolution_20261028"
    manifest = json.loads((ROOT / "datasets/ureter_semis_a1_sup_kidney_only_20261027/manifest.json").read_text(encoding="utf-8"))
    comparison = json.loads((output / "native_comparison/comparison.json").read_text(encoding="utf-8"))
    details = comparison["models"]
    frames = details["unet256"]["frames"]
    smallest = sorted([i for i, row in enumerate(frames) if 0 < row["gt_stone_pixels"] <= 1024], key=lambda i: frames[i]["gt_stone_pixels"])
    negative = sorted([i for i, row in enumerate(frames) if row["gt_stone_pixels"] == 0], key=lambda i: max(details[n]["frames"][i]["stone_fp"] for n in ("unet256", "unet512")), reverse=True)
    fiber = sorted(range(len(frames)), key=lambda i: details["unet512"]["frames"][i]["fiber_fn"], reverse=True)
    selected = []
    for group, count in ((smallest, 3), (negative, 1), (fiber, 2)):
        selected.extend([i for i in group if i not in selected][:count])
    for i in fiber:
        if len(selected) >= 6: break
        if i not in selected: selected.append(i)
    canvas = Image.new("RGB", (1024, 30 + 300*len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 7), "Native400 masks (display scaled256): Raw | Native GT | U-Net input256 | U-Net input512", fill="black")
    for j, i in enumerate(selected):
        row = manifest["val"][i]
        raw = np.asarray(Image.open(row["image"]).convert("RGB"))
        target = read_class_mask(row["mask"])
        labels = [None, target] + [np.asarray(Image.open(output / "native_comparison" / name / f"{i:04d}.png")) for name in ("unet256", "unet512")]
        y = 30 + j*300
        draw.text((4, y+3), f"{row['sample_id']} | Native GT stone pixels={frames[i]['gt_stone_pixels']}", fill="black")
        for column, label in enumerate(labels):
            panel = raw.copy()
            if label is not None:
                for c, color in ((1, [240, 40, 40]), (2, [0, 220, 240])):
                    region = label == c
                    panel[region] = (.45*panel[region]+.55*np.asarray(color)).astype(np.uint8)
            canvas.paste(Image.fromarray(panel).resize((256,256), Image.Resampling.BILINEAR), (column*256, y+23))
            if column >= 2:
                error = details[("unet256", "unet512")[column-2]]["frames"][i]
                draw.text((column*256+3, y+281), f"Stone TP/FP {error['stone_tp']}/{error['stone_fp']} Fiber FN {error['fiber_fn']}", fill="black")
    canvas.save(output / "native_failure_examples.png")
    atomic_json(output / "native_failure_examples.json", {"selection": "up to3 smallest<=1024 native-area frames,1 negative,2 high512fiber-FN frames, deduplicated",
        "sample_ids": [manifest["val"][i]["sample_id"] for i in selected], "evaluation_grid": [400,400], "display_tile": [256,256]})
