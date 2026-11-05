from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch

from tools.fiber_photometric_input import infer
from tools.fiber_reference_workspace import ROOT, read, sha256
from tools.fiber_tapered_tangent_cap import TaperedTangentCap
from tools.fiber_weak_stage2 import GrowthPipeline
from tools.g0_review_video import ReviewVideoWriter
from tools.member_experts import MemberExperts


MODEL_MANIFEST = ROOT / "models" / "manifest.json"
STAGE_CONFIG = ROOT / "configs" / "preprocess.json"
T1_CONFIG = ROOT / "configs" / "temporal.json"
G0_CONFIG = ROOT / "configs" / "g0.json"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_models(device: str) -> MemberExperts:
    manifest = read(MODEL_MANIFEST)
    stone = ROOT / "models" / manifest["stone_checkpoint"]["file"]
    fiber = ROOT / "models" / manifest["fiber_checkpoint"]["file"]
    for path, expected in [
        (stone, manifest["stone_checkpoint"]["sha256"]),
        (fiber, manifest["fiber_checkpoint"]["sha256"]),
    ]:
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen model missing or changed: {path}")
    experts = MemberExperts(stone, fiber, "model1", device=device)
    if experts.model_count != manifest["runtime_network_count"]:
        raise ValueError("Runtime network count changed")
    if experts.parameter_count != manifest["runtime_parameter_count"]:
        raise ValueError("Runtime parameter count changed")
    return experts


def prepare_output(path: Path) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    (path / "masks").mkdir(parents=True, exist_ok=True)
    return path


@torch.inference_mode()
def segment_video(input_path: Path, output: Path, device: str,
                  end_frame: int | None, warmup: int) -> dict:
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output = prepare_output(output)
    seed_everything()
    experts = load_models(device)
    stage = GrowthPipeline(read(STAGE_CONFIG), "weak_blue_extension_tip",
                           read(T1_CONFIG), sequential=True)
    g0 = TaperedTangentCap(read(G0_CONFIG))

    cap = cv2.VideoCapture(str(input_path))
    ok, first = cap.read()
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot decode input video: {input_path}")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Input video has no valid frame rate")

    for index in range(warmup):
        raw, _, fiber_probability, _ = infer(experts, first, "raw2")
        stage.run(first, raw, fiber_probability, index / fps, input_path.name)
    stage.reset()

    cap = cv2.VideoCapture(str(input_path))
    target_frames = min(source_frames, end_frame) if end_frame is not None else source_frames
    review = ReviewVideoWriter(
        output / "comparison.mp4", fps, target_frames, input_path.stem
    )
    records = []
    started = time.perf_counter()
    index = 0
    completed = False
    try:
        while end_frame is None or index < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            frame_started = time.perf_counter()
            raw, _, fiber_probability, network_timing = infer(experts, frame, "raw2")
            stage.run(frame, raw, fiber_probability, index / fps, input_path.name)
            g0.run(frame, raw, fiber_probability, stage.final_mask)
            g0_pipeline_ms = (time.perf_counter() - frame_started) * 1000
            mask = g0.final_mask
            if mask.dtype != np.uint8 or mask.shape != frame.shape[:2]:
                raise ValueError(f"Invalid G0 mask at frame {index}")
            if not set(np.unique(mask)) <= {0, 1, 2}:
                raise ValueError(f"Unexpected labels at frame {index}")
            mask_path = output / "masks" / f"{index:06d}.png"
            if not cv2.imwrite(str(mask_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise OSError(f"Failed to save {mask_path}")
            frame_total_ms = (time.perf_counter() - frame_started) * 1000
            review_record = review.write(
                frame, raw, mask, g0.info, index, index / fps,
                float(network_timing["pipeline_total_ms"]), g0_pipeline_ms,
            )
            records.append({
                "frame_index": index,
                "timestamp_seconds": index / fps,
                "mask": str(mask_path.relative_to(output)).replace("\\", "/"),
                "mask_sha256": array_sha256(mask),
                "g0": g0.info,
                "timing_ms": {
                    "network": float(network_timing["pipeline_total_ms"]),
                    "total": frame_total_ms,
                },
                "review": review_record,
            })
            index += 1
        review.close()
        completed = True
    finally:
        cap.release()
        if not completed:
            review.abort()
        stage.reset()
        del experts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (output / "frames.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    review_manifest = review.summary()
    review_manifest.update({
        "sha256": sha256(output / review_manifest["file"]),
        "bytes": (output / review_manifest["file"]).stat().st_size,
    })
    manifest = {
        "pipeline": "G0-only formal fiber segmentation",
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "frames": index,
        "source_fps": fps,
        "mask_labels": {"0": "background", "1": "stone", "2": "fiber"},
        "models": read(MODEL_MANIFEST),
        "config_sha256": {
            "stage": sha256(STAGE_CONFIG),
            "temporal_observer": sha256(T1_CONFIG),
            "g0": sha256(G0_CONFIG),
        },
        "g0_policy": "constructible geometry always applies; raw stone pixels are clipped; no proposal fallback",
        "device": device,
        "warmup_frames": warmup,
        "comparison_video": review_manifest,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the single frozen G0 pipeline from a raw video to native label "
            "masks and a synchronized comparison video."
        )
    )
    parser.add_argument("input", type=Path, help="raw input video")
    parser.add_argument(
        "output", type=Path,
        help="new or empty directory for masks, records, manifest and comparison video",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="PyTorch device (default: CUDA when available, otherwise CPU)",
    )
    parser.add_argument("--end-frame", type=int, default=None,
                        help="exclusive end frame, intended only for bounded verification")
    parser.add_argument("--warmup", type=int, default=12)
    args = parser.parse_args()
    if args.end_frame is not None and args.end_frame <= 0:
        parser.error("--end-frame must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    result = segment_video(args.input, args.output, args.device,
                           args.end_frame, args.warmup)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
