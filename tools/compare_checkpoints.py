"""Compare frozen checkpoints on the same reviewed validation pixels only."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ureter_semis.data import UreterDataset, load_manifest
from ureter_semis.experiment import atomic_json, input_identity, require_training_access, runtime_identity, seed_training, sha256, source_identity
from ureter_semis.inference import SegmentationPredictor
from ureter_semis.trainer import validate


def compare_checkpoints(manifest_path: str, checkpoints: list[str], output_path: str,
                        device_name: str = "cuda:0", batch_size: int = 2, allow_unreviewed_experiment: bool = False) -> dict:
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    root, manifest = load_manifest(manifest_path)
    if manifest.get("training_ready") is not True:
        if not allow_unreviewed_experiment:
            raise ValueError("Comparison requires an explicitly reviewed training_ready manifest")
        require_training_access(manifest, True)
    if not manifest["val"] or len(checkpoints) < 2 or batch_size < 1:
        raise ValueError("Need nonempty validation, >=2 checkpoints and batch_size > 0")
    paths = [Path(p).resolve() for p in checkpoints]
    if len(set(paths)) != len(paths):
        raise ValueError("Checkpoints must have distinct paths")
    subset = {"train": {"labeled": [], "unlabeled": []}, "val": manifest["val"]}
    identity = input_identity(Path(manifest_path).resolve(), root, subset)
    seed_training(42)
    device = torch.device(device_name)
    rows, size = [], None
    for path in paths:
        predictor = SegmentationPredictor(path, device_name)
        if size is not None and predictor.size != size:
            raise ValueError("Different model grids: define a separate common-resolution protocol first")
        size = predictor.size
        dataset = UreterDataset(root, manifest["val"], size, "val")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = validate(predictor.models, loader, device)
        rows.append({"checkpoint": str(path), "checkpoint_sha256": sha256(path), "metrics": metrics})
        del predictor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    baseline = rows[0]["metrics"]
    for row in rows[1:]:
        row["difference_from_first"] = {
            key: row["metrics"][key] - baseline[key] if row["metrics"][key] is not None and baseline[key] is not None else None
            for key in ("stone_dice", "fiber_dice", "mean_dice", "stone_negative_fp_image_rate",
                        "stone_negative_fp_pixel_fraction")}
    summary = {"schema_version": 1,
        "scope": "same validation set; not an independent-case generalization test",
        "dataset_role": manifest.get("role"), "split": "val", "test_pixels_read": False,
        "allow_unreviewed_experiment": allow_unreviewed_experiment,
        "grid_size": size, "inputs": identity, "runtime": runtime_identity(device),
        "source_hashes": source_identity(), "checkpoints": rows,
        "preprocessing": "PIL RGB bilinear image / nearest mask resize, ImageNet normalization; matches trainer validation",
        "video_inference_preprocessing": "A0 video replay retains OpenCV interpolation; evaluate video masks separately",
        "negative_presence": "native mask or explicit reviewed absence, before resize; unknown excluded",
        "metrics": "global confusion; ignore255; undefined denominators null; foreground macro over defined overlaps"}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--allow-unreviewed-experiment", action="store_true")
    args = parser.parse_args()
    result = compare_checkpoints(args.manifest, args.checkpoints, args.output, args.device, args.batch_size, args.allow_unreviewed_experiment)
    for row in result["checkpoints"]:
        print(f"{row['checkpoint']}: mean_dice={row['metrics']['mean_dice']}")


if __name__ == "__main__":
    main()
