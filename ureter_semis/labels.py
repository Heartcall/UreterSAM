"""Read class-index masks without converting palette colors into class ids."""
from pathlib import Path

import numpy as np
from PIL import Image


def read_class_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image).copy()
    if array.ndim != 2 or not set(np.unique(array)).issubset({0, 1, 2, 255}):
        raise ValueError(f"Expected class-index PNG (0/1/2/255), not color rendering: {path}")
    return array.astype(np.uint8)
