"""Fail-fast checks for effective foreground/background supervision."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .labels import read_class_mask
from .source_policy import require_training_sources


def audit_supervision(root: Path, manifest: dict) -> dict:
    require_training_sources(manifest)
    training = manifest["train"]["labeled"] + manifest["train"].get("unlabeled", [])
    validation_hashes = {r["image_sha256"] for r in manifest.get("val", []) if r.get("image_sha256")}
    for row in training:
        if row.get("video_id") in {"video_1", "video_2"} or row.get("annotation_basis") == "reviewed_visible_pixels":
            raise ValueError("video_1 and video_2 are validation only; training video frames are forbidden")
        if row.get("image_sha256") in validation_hashes:
            raise ValueError("Exact validation image appears in training")
    total, sources, samples = Counter(), {}, []
    for row in manifest["train"]["labeled"]:
        mask = read_class_mask(root / row["mask"])
        labels, counts = np.unique(mask, return_counts=True)
        pixels = {int(label): int(count) for label, count in zip(labels, counts)}
        if set(pixels) - {0, 1, 2, 255}:
            raise ValueError("Unsupported supervised label")
        total.update(pixels)
        source = row.get("source_dataset", "unspecified")
        sources.setdefault(source, Counter()).update(pixels)
        samples.append({"sample_id": row.get("sample_id", row["image"]), "pixels": pixels})
    if total[0] == 0:
        raise ValueError("No background supervision: background pixel count is zero")
    if total[1] == 0 or total[2] == 0:
        raise ValueError("Both stone and fiber require effective supervised pixels")
    return {"pixel_counts": dict(total), "source_pixel_counts": {k: dict(v) for k, v in sources.items()},
            "labeled_images": len(samples), "samples": samples}
