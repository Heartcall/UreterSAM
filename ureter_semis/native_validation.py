"""PIL input resize and probability-first evaluation against untouched native GT."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .data import _tensor
from .labels import read_class_mask
from .metrics import confusion_matrix, metrics_from_confusion
from .model import ensemble_logits


class NativeValidationDataset(Dataset):
    def __init__(self, root: Path, samples: list[dict], size: int):
        self.root, self.samples, self.size = root, samples, size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        row = self.samples[i]
        image = Image.open(self.root / row["image"]).convert("RGB")
        target = read_class_mask(self.root / row["mask"])
        if target.shape != (image.height, image.width):
            raise ValueError("Native validation image/mask coordinate mismatch")
        presence = row.get("stone_presence_review", "unknown")
        if presence == "unknown":
            presence = "present" if (target == 1).any() else ("absent" if (target != 255).all() else "unknown")
        return {"image": _tensor(image.resize((self.size, self.size), Image.Resampling.BILINEAR)),
                "target": torch.from_numpy(target.astype(np.int64).copy()),
                "stone_presence": presence, "sample_id": row.get("sample_id", str(i))}


def collate_native(rows):
    return {"image": torch.stack([r["image"] for r in rows]),
            "targets": [r["target"] for r in rows],
            "stone_presence": [r["stone_presence"] for r in rows],
            "sample_ids": [r["sample_id"] for r in rows]}


def native_argmax(probability: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    if probability.ndim != 3 or probability.shape[0] != 3:
        raise ValueError("Expected three-class probability [3,H,W]")
    return F.interpolate(probability.unsqueeze(0).float(), size=shape, mode="bilinear",
                         align_corners=False)[0].argmax(0)


@torch.inference_mode()
def validate_native(models, loader, device, collect_frames=False):
    for model in models: model.eval()
    matrix = torch.zeros((3, 3), dtype=torch.long, device=device)
    negatives = fp_images = negative_pixels = fp_pixels = 0
    frames = []
    for batch in loader:
        probabilities = ensemble_logits(models, batch["image"].to(device)).softmax(1)
        for i, target_cpu in enumerate(batch["targets"]):
            target = target_cpu.to(device)
            prediction = native_argmax(probabilities[i], tuple(target.shape))
            matrix += confusion_matrix(prediction, target)
            valid = target != 255
            if batch["stone_presence"][i] == "absent" and valid.any():
                false_stone = (prediction == 1) & valid
                negatives += 1; fp_images += int(false_stone.any())
                negative_pixels += int(valid.sum()); fp_pixels += int(false_stone.sum())
            if collect_frames:
                frames.append({"sample_id": batch["sample_ids"][i],
                    "prediction": prediction.byte().cpu().numpy(), "target": target_cpu.numpy(),
                    "stone_presence": batch["stone_presence"][i]})
    if not int(matrix.sum()): raise ValueError("Native validation has no valid pixels")
    metrics = metrics_from_confusion(matrix)
    metrics.update({"stone_negative_images": negatives, "stone_negative_images_with_fp": fp_images,
        "stone_negative_fp_image_rate": fp_images / negatives if negatives else None,
        "stone_negative_valid_pixels": negative_pixels, "stone_negative_fp_pixels": fp_pixels,
        "stone_negative_fp_pixel_fraction": fp_pixels / negative_pixels if negative_pixels else None})
    return (metrics, frames) if collect_frames else metrics
