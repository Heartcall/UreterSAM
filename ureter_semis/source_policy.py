"""Honor explicit dataset exclusions for supervised, unlabeled and reserved pools."""
from __future__ import annotations

import json
from pathlib import Path
import re


def excluded_training_source(row: dict, blocked_hashes: set[str] | None = None) -> bool:
    fields = ("source_dataset", "sample_id", "image", "mask", "source_image", "source_label", "source_mask")
    if any("stonenew" in re.sub(r"[^a-z0-9]", "", str(row.get(key, "")).lower()) for key in fields):
        return True
    return bool(blocked_hashes and row.get("image_sha256") in blocked_hashes)


def require_training_sources(manifest: dict) -> None:
    root = Path(__file__).resolve().parents[1]
    registry_path = root / "annotations/excluded_training_sources_20261027.json"
    blocked_hashes = set()
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        blocked_hashes = set(registry["excluded_image_sha256"])
    train = manifest.get("train", {})
    for split, rows in (("labeled", train.get("labeled", [])), ("unlabeled", train.get("unlabeled", [])),
                        ("reserved_unlabeled", manifest.get("reserved_unlabeled", []))):
        for row in rows:
            if excluded_training_source(row, blocked_hashes):
                raise ValueError(f"stone_new is excluded from training by user instruction (fiber-tip-only annotations); "
                                 f"remove sample from {split}: {row.get('sample_id', row.get('image'))}")
