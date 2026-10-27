from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from .labels import read_class_mask


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def _tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy((array - MEAN) / STD)


def _geometric(image: Image.Image, mask: Image.Image | None) -> tuple[Image.Image, Image.Image | None]:
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if mask is not None else None
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM) if mask is not None else None
    return image, mask


def _strong(image: Image.Image) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
    image = ImageEnhance.Color(image).enhance(random.uniform(0.75, 1.25))
    if random.random() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5)))
    return image


class UreterDataset(Dataset):
    def __init__(self, root: Path, samples: list[dict], size: int, mode: str) -> None:
        self.root = root
        self.samples = samples
        self.size = (size, size)
        self.mode = mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image_path = self.root / sample["image"]
        image = Image.open(image_path).convert("RGB").resize(self.size, Image.Resampling.BILINEAR)

        if self.mode == "unlabeled":
            weak, _ = _geometric(image, None)
            return {"weak": _tensor(weak), "strong": _tensor(_strong(weak.copy())), "name": str(image_path)}

        mask_path = self.root / sample["mask"]
        mask = Image.fromarray(read_class_mask(mask_path)).resize(self.size, Image.Resampling.NEAREST)
        if self.mode == "train":
            image, mask = _geometric(image, mask)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy()).long()
        return {"image": _tensor(image), "mask": mask_tensor, "name": str(image_path)}


def load_manifest(path: str | Path) -> tuple[Path, dict]:
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    root = Path(manifest["root"]).expanduser()
    if not root.is_absolute():
        root = manifest_path.parent / root
    return root.resolve(), manifest


def prepare_roboflow(source: str | Path, output: str | Path, labeled_fraction: float = 0.2, seed: int = 42) -> Path:
    """Backward-compatible wrapper; conversion lives in the standalone tools module."""
    from tools.convert_dataset import prepare_dataset

    return prepare_dataset(source, output, format="yolo", labeled_fraction=labeled_fraction, seed=seed)
