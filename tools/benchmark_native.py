"""Fresh-process cost from cached BGR frame to native probability maps and masks."""
import argparse
import gc
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ureter_semis.data import load_manifest
from ureter_semis.experiment import atomic_json, seed_training, sha256
from ureter_semis.inference import SegmentationPredictor, preprocess
from ureter_semis.model import ensemble_logits
from pathlib import Path


def benchmark(manifest_path, checkpoint, output_path, warmup=12, repeats=3):
    output = Path(output_path).resolve()
    if output.exists(): raise FileExistsError(output)
    seed_training(42)
    root, manifest = load_manifest(manifest_path)
    frames = [cv2.imread(str(root / row["image"])) for row in manifest["val"]]
    if any(frame is None for frame in frames): raise ValueError("Missing benchmark frame")
    predictor = SegmentationPredictor(checkpoint)
    @torch.inference_mode()
    def infer(frame):
        predictor._sync(); start = time.perf_counter()
        image = preprocess(frame, predictor.size, predictor.device)
        predictor._sync(); prepared = time.perf_counter()
        logits = ensemble_logits(predictor.models, image)
        predictor._sync(); inferred = time.perf_counter()
        probability = F.interpolate(logits.softmax(1), size=frame.shape[:2], mode="bilinear", align_corners=False)
        mask = probability.argmax(1).byte().cpu().numpy()
        array = probability[0].cpu().numpy()
        predictor._sync(); finished = time.perf_counter()
        return {"pipeline_ms": (finished-start)*1000, "preprocess_ms": (prepared-start)*1000,
                "model_ms": (inferred-prepared)*1000, "native_postprocess_ms": (finished-inferred)*1000}
    for i in range(warmup): infer(frames[i % len(frames)])
    torch.cuda.reset_peak_memory_stats()
    measurements = [infer(frame) for _ in range(repeats) for frame in frames]
    result = {"checkpoint_sha256": sha256(Path(checkpoint)), "input_size": predictor.size,
        "model_count": predictor.model_count, "parameter_count": sum(p.numel() for m in predictor.models for p in m.parameters()),
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "warmup_frames": warmup, "samples": len(measurements),
        "scope": "Fresh process, batch1 cached native BGR via OpenCV to native probability maps/masks; excludes decode/display/geometry/memory; accuracy uses PIL separately",
        "timings": {k: {"mean": float(np.mean([r[k] for r in measurements])), "p50": float(np.percentile([r[k] for r in measurements],50)),
                         "p95": float(np.percentile([r[k] for r in measurements],95))} for k in measurements[0]}, "raw_samples": measurements}
    del predictor
    gc.collect(); torch.cuda.empty_cache()
    result["post_release_cuda_allocated_mib"] = torch.cuda.memory_allocated() / 2**20
    atomic_json(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(); benchmark(args.manifest, args.checkpoint, args.output, args.warmup, args.repeats)
