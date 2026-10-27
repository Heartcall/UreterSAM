"""Run reproducible A0 video inference and export unlabelled review candidates.

Usage: python -m tools.evaluate_video --config runs/baselines/a0_20261027/config.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from ureter_semis.inference import SegmentationPredictor, file_sha256
from ureter_semis.video_metrics import estimated_fov, prediction_diagnostics


def save_png(path: Path, array: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        raise RuntimeError(f"PNG encoding failed: {path}")
    encoded.tofile(str(path))


def render(frame: np.ndarray, mask: np.ndarray, frame_index: int) -> np.ndarray:
    labels = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = frame.copy()
    for cls, color in ((1, (40, 40, 235)), (2, (235, 210, 20))):
        selected = labels == cls
        overlay[selected] = np.clip(frame[selected]*0.45 + np.array(color)*0.55, 0, 255).astype(np.uint8)
    canvas = np.concatenate([frame, overlay], axis=1)
    cv2.putText(canvas, f"A0 raw prediction | frame {frame_index} | NOT reviewed ground truth",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def propose_clips(rows: list[dict], fps: float, max_clips: int = 6) -> list[dict]:
    length = max(1, round(4*fps))
    proposed = []
    ranked = []
    for i, row in enumerate(rows):
        area = row["stone_fraction_estimated_fov"] or 0
        prior = rows[i-1]["stone_fraction_estimated_fov"] or 0 if i else area
        change = row["uncompensated_label_change_fraction"] or 0
        score = abs(area-prior)*2 + change + (0.2 if area > 0.25 else 0)
        ranked.append((score, i))
    for score, center in sorted(ranked, reverse=True):
        start = max(0, min(center-length//2, len(rows)-length))
        end = min(len(rows), start+length)
        if any(start < p["end_frame_exclusive"] and end > p["start_frame"] for p in proposed):
            continue
        proposed.append({"start_frame": start, "end_frame_exclusive": end,
                         "center_frame": center, "start_s": start/fps, "end_s": end/fps,
                         "proposal_score": score, "review_status": "pending",
                         "event_labels": [], "ground_truth_available": False,
                         "reason": "prediction size/change diagnostic; NOT a verified failure"})
        if len(proposed) == max_clips:
            break
    return sorted(proposed, key=lambda p: p["start_frame"])


def run_video(predictor: SegmentationPredictor, item: dict, output: Path,
              config: dict, save_probabilities: bool = True) -> dict:
    source = Path(item["path"])
    if file_sha256(source) != item["sha256"]:
        raise ValueError(f"Input hash differs from frozen config: {source}")
    output.mkdir(parents=True, exist_ok=False)
    masks = output / "pred_masks"
    masks.mkdir()
    probability_dir = output / "probabilities"
    if save_probabilities:
        probability_dir.mkdir()
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(3)), int(cap.get(4))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Valid frame rate required for event timing")
    writer = cv2.VideoWriter(str(output / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width*2, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create preview video")
    previous = None
    rows, probability_buffer, indices = [], [], []
    repeat_verified = False
    started = time.perf_counter()
    try:
        with (output / "frames.jsonl").open("w", encoding="utf-8") as log:
            while True:
                t0 = time.perf_counter()
                ok, frame = cap.read()
                read_ms = (time.perf_counter()-t0)*1000
                if not ok:
                    break
                if not rows:
                    for _ in range(3):
                        predictor.predict(frame)
                    a, b = predictor.predict(frame), predictor.predict(frame)
                    repeat_verified = np.array_equal(a.mask, b.mask) and np.array_equal(a.probabilities, b.probabilities)
                    if not repeat_verified:
                        raise RuntimeError("Identical-frame inference is not bitwise reproducible")
                begin = time.perf_counter()
                result = predictor.predict(frame)
                index = len(rows)
                diagnostics = prediction_diagnostics(result.mask, result.probabilities,
                                                     estimated_fov(frame, result.mask.shape), previous)
                save_png(masks / f"{index:06d}.png", result.mask)
                if save_probabilities:
                    probability_buffer.append(result.probabilities.astype(np.float16))
                    indices.append(index)
                    if len(indices) == 64:
                        np.savez_compressed(probability_dir / f"{indices[0]:06d}.npz",
                                            frame_indices=np.array(indices), probabilities=np.stack(probability_buffer))
                        probability_buffer.clear()
                        indices.clear()
                writer.write(render(frame, result.mask, index))
                elapsed = (time.perf_counter()-begin)*1000
                row = {"video_id": item["video_id"], "frame_index": index,
                       "timestamp_s": index/fps, "coordinate_system": "raw_model_grid",
                       "ground_truth_available": False, **diagnostics,
                       "read_ms": read_ms, "preprocess_ms": result.preprocess_ms,
                       "model_ms": result.model_ms, "postprocess_ms": result.postprocess_ms,
                       "processing_ms_including_exports": elapsed,
                       "latency_ms_including_read_and_exports": read_ms+elapsed}
                log.write(json.dumps(row, allow_nan=False)+"\n")
                rows.append(row)
                previous = result.mask
                if len(rows) % 200 == 0:
                    print(f"{item['video_id']}: {len(rows)} frames", flush=True)
            if indices:
                np.savez_compressed(probability_dir / f"{indices[0]:06d}.npz",
                                    frame_indices=np.array(indices), probabilities=np.stack(probability_buffer))
    finally:
        cap.release()
        writer.release()
    if not rows:
        raise RuntimeError("Empty input video")
    clips = propose_clips(rows, fps)
    review = {"video_id": item["video_id"], "patient_id": item.get("patient_id"),
              "source_group": item.get("source_group"), "role": "development_only",
              "source": str(source), "source_sha256": item["sha256"], "fps": fps,
              "clips": clips, "annotation_schema": "docs/evaluation/video_annotation_protocol.md"}
    (output / "review_candidates.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
    cap = cv2.VideoCapture(str(source))
    try:
        for ci, clip in enumerate(clips):
            clip_dir = output / "review_clips"
            clip_dir.mkdir(exist_ok=True)
            cap.set(cv2.CAP_PROP_POS_FRAMES, clip["start_frame"])
            clip_writer = cv2.VideoWriter(str(clip_dir / f"clip_{ci:02d}.mp4"),
                                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (width*2, height))
            if not clip_writer.isOpened():
                raise RuntimeError("Cannot create review clip")
            try:
                for idx in range(clip["start_frame"], clip["end_frame_exclusive"]):
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("Unexpected end of video during clip export")
                    mask = cv2.imread(str(masks / f"{idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
                    canvas = render(frame, mask, idx)
                    clip_writer.write(canvas)
                    if idx == clip["center_frame"]:
                        save_png(clip_dir / f"clip_{ci:02d}_center.png", canvas)
            finally:
                clip_writer.release()
    finally:
        cap.release()
    timing = {}
    for key in ("preprocess_ms", "model_ms", "postprocess_ms", "processing_ms_including_exports",
                "latency_ms_including_read_and_exports"):
        values = [r[key] for r in rows]
        timing[key] = {"mean": float(np.mean(values)), "p50": float(np.median(values)),
                       "p95": float(np.percentile(values, 95))}
    summary = {"video_id": item["video_id"], "experiment_id": config["experiment_id"],
               "frames": len(rows), "source_fps": fps, "role": "development_only",
               "ground_truth_available": False, "accuracy_metrics": None,
               "repeat_check_bitwise_equal": repeat_verified, "timing": timing,
               "wall_s_including_warmup_and_review_exports": time.perf_counter()-started,
               "large_region_candidate_frames": sum((r["stone_fraction_estimated_fov"] or 0)>config["large_region_fraction"] for r in rows),
               "review_clips": len(clips), "probabilities_saved": save_probabilities,
               "notes": ["Candidate counts are NOT false-positive counts.",
                         "Label changes are NOT motion compensated and are diagnostic only.",
                         "FOV is estimated, not manually annotated.",
                         "Final partial probability-chunk flush and JSONL writes are excluded from per-frame latency; included in wall time."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="default: a new predictions directory beside config")
    parser.add_argument("--no-probabilities", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("precision") != "float32" or config.get("temporal") is not False or config.get("camera_correction") is not False:
        raise ValueError("This A0 runner only supports raw float32, non-temporal segmentation")
    output = Path(args.output).resolve() if args.output else config_path.parent / "predictions"
    output.mkdir(parents=True, exist_ok=False)
    runtime_sources = {str(p): file_sha256(p) for p in (
        Path(__file__), Path("ureter_semis/inference.py"), Path("ureter_semis/video_metrics.py"),
        Path("ureter_semis/model.py"), Path("ureter_semis/data.py"))}
    (output / "run_provenance.json").write_text(json.dumps({"config_path": str(config_path),
        "config_sha256": file_sha256(config_path), "runner_sources": runtime_sources,
        "probabilities_saved": not args.no_probabilities}, indent=2), encoding="utf-8")
    if file_sha256(config["checkpoint"]) != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from frozen config")
    predictor = SegmentationPredictor(config["checkpoint"], config["device"], config["seed"])
    if predictor.size != config["size"]:
        raise ValueError("Checkpoint size differs from config")
    summaries = [run_video(predictor, item, output / item["video_id"], config,
                           not args.no_probabilities) for item in config["inputs"]]
    (output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Completed: {output}")


if __name__ == "__main__":
    main()
