
"""Compare predicted fiber geometry with annotation geometry."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from ureter_semis.data import load_manifest
from ureter_semis.demo import _fiber_geometry, _fiber_ray


root, manifest = load_manifest(
    "datasets/ureter_semis/manifest.json"
)

pred_dir = Path("runs/vis_test/pred_masks")
output = Path("runs/geometry_test/fiber_quality.csv")
output.parent.mkdir(parents=True, exist_ok=True)

rows = []

for sample in manifest["test"]:
    name = Path(sample["image"]).stem
    pred_path = pred_dir / f"{name}.png"

    if not pred_path.exists():
        print("Missing:", pred_path)
        continue

    # Prediction mask: 0=background, 1=stone, 2=fiber
    pred = np.asarray(
        Image.open(pred_path).convert("L"),
        dtype=np.uint8,
    )

    h, w = pred.shape

    # Resize GT mask to match prediction resolution
    truth = np.asarray(
        Image.open(root / sample["mask"])
        .convert("L")
        .resize((w, h), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )

    gt_present = bool((truth == 2).any())
    pred_present = bool((pred == 2).any())

    gt_geometry = _fiber_geometry(truth == 2)
    pred_geometry = _fiber_geometry(pred == 2)

    gt_width = (
        gt_geometry["width_px"]
        if gt_geometry is not None else None
    )

    pred_width = (
        pred_geometry["width_px"]
        if pred_geometry is not None else None
    )

    width_error = (
        abs(gt_width - pred_width)
        if gt_width is not None and pred_width is not None
        else None
    )

    axis_error = None

    if gt_geometry is not None and pred_geometry is not None:
        gt_ray = _fiber_ray(gt_geometry["component"])
        pred_ray = _fiber_ray(pred_geometry["component"])

        if gt_ray is not None and pred_ray is not None:
            a = np.asarray(gt_ray["direction"])
            b = np.asarray(pred_ray["direction"])

            # Axes have a 180-degree sign ambiguity.
            cosine = np.clip(abs(np.dot(a, b)), 0, 1)

            axis_error = float(
                np.degrees(np.arccos(cosine))
            )

    rows.append({
        "image": Path(sample["image"]).name,
        "gt_fiber_present": gt_present,
        "pred_fiber_present": pred_present,
        "gt_width_px": gt_width,
        "pred_width_px": pred_width,
        "abs_width_error_px": width_error,
        "axis_error_deg": axis_error,
    })


if not rows:
    raise RuntimeError("No prediction masks processed")


with output.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(rows)


false_positive = [
    r for r in rows
    if not r["gt_fiber_present"] and r["pred_fiber_present"]
]

missed = [
    r for r in rows
    if r["gt_fiber_present"] and not r["pred_fiber_present"]
]

width_errors = [
    r["abs_width_error_px"]
    for r in rows
    if r["abs_width_error_px"] is not None
]

axis_errors = [
    r["axis_error_deg"]
    for r in rows
    if r["axis_error_deg"] is not None
]


print("\n=== Fiber Geometry Quality ===")
print("Images processed:", len(rows))

print(
    "GT fiber images:",
    sum(r["gt_fiber_present"] for r in rows),
)

print(
    "Predicted fiber images:",
    sum(r["pred_fiber_present"] for r in rows),
)

print(
    "False-positive fiber images:",
    len(false_positive),
)

print(
    "Missed fiber images:",
    len(missed),
)

print(
    "False-positive examples:",
    [r["image"] for r in false_positive[:10]],
)

print(
    "Missed examples:",
    [r["image"] for r in missed[:10]],
)

if width_errors:
    print(
        "Fiber width MAE:",
        f"{np.mean(width_errors):.2f} px",
    )

if axis_errors:
    print(
        "Fiber axis MAE:",
        f"{np.mean(axis_errors):.2f} degrees",
    )

print("Results saved to:", output)
