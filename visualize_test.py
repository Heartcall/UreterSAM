
"""Visualize UreterSAM predictions and evaluate individual images."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from ureter_semis.data import UreterDataset, load_manifest
from ureter_semis.model import build_model


# RGB colors: stone = red, fiber = cyan
COLORS = {
    1: (255, 50, 50),
    2: (0, 225, 235),
}


def blend_mask(image, mask, alpha=0.45):
    """Overlay a semantic mask on an RGB image."""
    result = np.asarray(image, dtype=np.uint8).copy()

    for class_id, color in COLORS.items():
        selected = mask == class_id

        if np.any(selected):
            pixels = result[selected].astype(np.float32)
            result[selected] = np.clip(
                pixels * (1 - alpha)
                + np.asarray(color) * alpha,
                0,
                255,
            ).astype(np.uint8)

    return Image.fromarray(result)


def make_panel(image, truth, prediction):
    """Create a 2x2 comparison panel."""
    original = np.asarray(image, dtype=np.uint8)

    error = original.copy()
    wrong = truth != prediction

    # Highlight incorrect pixels in magenta.
    if np.any(wrong):
        error[wrong] = np.clip(
            original[wrong].astype(np.float32) * 0.3
            + np.asarray((255, 0, 255)) * 0.7,
            0,
            255,
        ).astype(np.uint8)

    views = [
        image,
        blend_mask(image, truth),
        blend_mask(image, prediction),
        Image.fromarray(error),
    ]

    labels = [
        "Original",
        "Ground truth",
        "Prediction",
        "Errors (magenta)",
    ]

    width, height = image.size
    header = 28

    panel = Image.new(
        "RGB",
        (width * 2, (height + header) * 2),
        (20, 20, 20),
    )

    draw = ImageDraw.Draw(panel)

    for i, (view, label) in enumerate(zip(views, labels)):
        x = (i % 2) * width
        y = (i // 2) * (height + header)

        draw.text((x + 8, y + 7), label, fill="white")
        panel.paste(view, (x, y + header))

    return panel


def class_scores(cm, class_id):
    """Calculate Dice and IoU from a confusion matrix."""
    tp = int(cm[class_id, class_id])
    fp = int(cm[:, class_id].sum()) - tp
    fn = int(cm[class_id, :].sum()) - tp

    denominator = 2 * tp + fp + fn

    # Undefined if the class is absent in both masks.
    dice = 2 * tp / denominator if denominator else None

    union = tp + fp + fn
    iou = tp / union if union else None

    return dice, iou, tp, fp, fn


def main():
    parser = argparse.ArgumentParser(
        description="UreterSAM segmentation visualization"
    )

    parser.add_argument(
        "--checkpoint",
        default="runs/endosemis/best.pt",
    )
    parser.add_argument(
        "--manifest",
        default="datasets/ureter_semis/manifest.json",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
    )
    parser.add_argument(
        "--output",
        default="runs/vis_test",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 means process all images",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; try --device cpu"
        )

    # Load the best checkpoint.
    # Only load checkpoints from trusted sources.
    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    size = int(ckpt.get("size", 256))

    # Build the two U-Net models.
    models = [
        build_model(pretrained=False).to(device).eval()
        for _ in range(2)
    ]

    models[0].load_state_dict(ckpt["model1"])
    models[1].load_state_dict(ckpt["model2"])

    for model in models:
        model.eval()

    # Load the dataset.
    root, manifest = load_manifest(args.manifest)

    samples = manifest[args.split]

    if args.limit > 0:
        samples = samples[:args.limit]

    if not samples:
        raise RuntimeError(
            "No images found in the selected split"
        )

    dataset = UreterDataset(
        root,
        samples,
        size,
        "val",
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # Prepare output directories.
    output = Path(args.output)
    panels = output / "panels"
    masks = output / "pred_masks"

    panels.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)

    total_cm = torch.zeros(
        (3, 3),
        dtype=torch.long,
    )

    rows = []

    # Run inference.
    with torch.inference_mode():
        for index, batch in enumerate(loader, start=1):

            image_tensor = batch["image"].to(device)

            truth = (
                batch["mask"][0]
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            # Average the logits of both U-Nets.
            logits = (
                models[0](image_tensor)
                + models[1](image_tensor)
            ) / 2

            prediction = (
                logits.argmax(dim=1)[0]
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            # Accumulate the pixel confusion matrix.
            cm = torch.bincount(
                torch.from_numpy(
                    (
                        truth.astype(np.int64) * 3
                        + prediction.astype(np.int64)
                    ).ravel()
                ),
                minlength=9,
            ).reshape(3, 3)

            total_cm += cm

            # Load the original image at model resolution.
            image_path = Path(batch["name"][0])

            image = (
                Image.open(image_path)
                .convert("RGB")
                .resize(
                    (size, size),
                    Image.Resampling.BILINEAR,
                )
            )

            name = image_path.stem

            # Save a 2x2 visualization.
            make_panel(
                image,
                truth,
                prediction,
            ).save(
                panels / f"{name}.png"
            )

            # Save semantic prediction: 0, 1, 2.
            Image.fromarray(prediction).save(
                masks / f"{name}.png"
            )

            # Calculate metrics for this image.
            stone_dice, _, _, _, _ = class_scores(cm, 1)
            fiber_dice, _, _, _, _ = class_scores(cm, 2)

            row = {
                "image": image_path.name,
                "stone_dice": stone_dice,
                "fiber_dice": fiber_dice,
                "stone_gt_pixels": int((truth == 1).sum()),
                "stone_pred_pixels": int((prediction == 1).sum()),
                "fiber_gt_pixels": int((truth == 2).sum()),
                "fiber_pred_pixels": int((prediction == 2).sum()),
                "error_pixels": int(
                    (truth != prediction).sum()
                ),
            }

            rows.append(row)

            print(
                f"[{index}/{len(dataset)}] {name}: "
                f"stone={stone_dice if stone_dice is not None else 'N/A'}, "
                f"fiber={fiber_dice if fiber_dice is not None else 'N/A'}"
            )

    # Save per-image metrics as CSV.
    fields = list(rows[0].keys())

    with (
        output / "per_image.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    # Calculate overall dataset metrics.
    report = {
        "split": args.split,
        "images": len(rows),
        "checkpoint": args.checkpoint,
        "size": size,
        "classes": {},
        "confusion_matrix": total_cm.tolist(),
    }

    for class_id, label in [
        (1, "stone"),
        (2, "fiber"),
    ]:
        dice, iou, tp, fp, fn = class_scores(
            total_cm,
            class_id,
        )

        report["classes"][label] = {
            "dice": dice,
            "iou": iou,
            "precision": (
                tp / (tp + fp)
                if tp + fp else None
            ),
            "recall": (
                tp / (tp + fn)
                if tp + fn else None
            ),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "images_with_class": sum(
                r[f"{label}_gt_pixels"] > 0
                for r in rows
            ),
        }

    report["mean_dice"] = sum(
        report["classes"][c]["dice"]
        for c in ("stone", "fiber")
    ) / 2

    report["mean_iou"] = sum(
        report["classes"][c]["iou"]
        for c in ("stone", "fiber")
    ) / 2

    (output / "summary.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Identify the images most worth inspecting.
    with (
        output / "inspect_first.txt"
    ).open("w", encoding="utf-8") as handle:

        handle.write(
            "Worst 10 images that contain ground-truth fiber:\n"
        )

        with_fiber = [
            r for r in rows
            if r["fiber_gt_pixels"] > 0
        ]

        for r in sorted(
            with_fiber,
            key=lambda x: x["fiber_dice"],
        )[:10]:
            handle.write(
                f"{r['image']}  "
                f"fiber_dice={r['fiber_dice']:.4f}\n"
            )

        handle.write(
            "\nFalse positives on images "
            "WITHOUT ground-truth fiber:\n"
        )

        false_positives = [
            r for r in rows
            if r["fiber_gt_pixels"] == 0
            and r["fiber_pred_pixels"] > 0
        ]

        for r in sorted(
            false_positives,
            key=lambda x: -x["fiber_pred_pixels"],
        )[:10]:
            handle.write(
                f"{r['image']}  "
                f"predicted_fiber_pixels="
                f"{r['fiber_pred_pixels']}\n"
            )

    # Print overall metrics.
    print("\n=== Overall ===")

    for label in ("stone", "fiber"):
        m = report["classes"][label]

        print(
            f"{label}: "
            f"Dice={m['dice']:.4f}, "
            f"IoU={m['iou']:.4f}, "
            f"Precision={m['precision']:.4f}, "
            f"Recall={m['recall']:.4f}"
        )

    print(
        f"Mean Dice={report['mean_dice']:.4f}, "
        f"Mean IoU={report['mean_iou']:.4f}"
    )

    print(
        f"Results saved to: {output.resolve()}"
    )


if __name__ == "__main__":
    main()
