"""Check real dual-U-Net resume on synthetic images; NOT an A1 experiment."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ureter_semis.experiment import atomic_json
from ureter_semis.inference import SegmentationPredictor
from ureter_semis.trainer import train_endosemis


def compare(left, right, path="checkpoint") -> int:
    if isinstance(left, torch.Tensor):
        if not torch.equal(left, right):
            raise AssertionError(f"Tensor differs: {path}")
        return 1
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"Keys differ: {path}")
        return sum(compare(left[k], right[k], f"{path}.{k}") for k in left)
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"Lengths differ: {path}")
        return sum(compare(a, b, f"{path}[{i}]") for i, (a, b) in enumerate(zip(left, right)))
    if left != right:
        raise AssertionError(f"Values differ: {path}: {left!r} vs {right!r}")
    return 0


def run(output: str | Path, device: str = "cuda:0", size: int = 64) -> dict:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    data = output / "synthetic_data"
    data.mkdir(parents=True)
    rng = np.random.default_rng(835)
    rows = []
    for i in range(6):
        image = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), np.uint8)
        if i % 2:
            mask[8:size//2, 8:size//2] = 1
        mask[-8:-4, 5:-5] = 2
        mask[:2] = 255
        Image.fromarray(image).save(data / f"image_{i}.png")
        Image.fromarray(mask).save(data / f"mask_{i}.png")
        rows.append({"image": f"image_{i}.png", "mask": f"mask_{i}.png",
                     "stone_presence_review": "present" if i % 2 else "absent"})
    manifest = data / "manifest.json"
    atomic_json(manifest, {"root": str(data), "training_ready": True,
        "role": "synthetic_engineering_test_not_medical_data",
        "train": {"labeled": rows[:2], "unlabeled": rows[2:4]}, "val": rows[4:], "test": []})
    args = {"manifest_path": str(manifest), "epochs": 2, "size": size, "batch_size": 2,
            "device_name": device, "pretrained": False, "seed": 42}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_endosemis(**args, output_dir=str(output / "full"))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_endosemis(**args, output_dir=str(output / "staged"), stop_after_epoch=1)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_endosemis(**args, output_dir=str(output / "resumed"), resume=str(output / "staged/last.pt"))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    a = torch.load(output / "full/last.pt", map_location="cpu", weights_only=True)
    b = torch.load(output / "resumed/last.pt", map_location="cpu", weights_only=True)
    tensor_count = compare(a, b)
    if any(row["optimizer_update_steps"] != 1 for row in a["history"]):
        raise AssertionError("Smoke check must include actual optimizer updates")
    runtime, history = a["run_identity"]["runtime"], a["history"]
    del a, b
    gc.collect()
    best_tensors = compare(torch.load(output / "full/best.pt", map_location="cpu", weights_only=True),
                           torch.load(output / "resumed/best.pt", map_location="cpu", weights_only=True))
    predictor = SegmentationPredictor(output / "resumed/best.pt", device=device)
    frame = np.zeros((size, size, 3), np.uint8)
    result = predictor.predict(frame)
    if result.mask.shape != (size, size) or not set(np.unique(result.mask)) <= {0,1,2}:
        raise AssertionError("New checkpoint incompatible with stateless inference")
    summary = {"scope": "synthetic engineering check; no medical accuracy or A1 claim",
               "size": size, "runtime": runtime, "full_vs_resume_bitwise_equal": True,
               "checkpoint_tensors_compared": tensor_count, "best_tensors_compared": best_tensors,
               "new_checkpoint_inference_passed": True, "epochs": history,
               "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}
    atomic_json(output / "summary.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k != "epochs"}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    run(args.output, args.device, args.size)


if __name__ == "__main__":
    main()
