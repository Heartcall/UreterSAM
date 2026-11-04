"""Side-by-side UreterSAM video demo with live FPS / frame number.

Run from project root:
python realtime_compare.py --source data/videos/video_1.mp4
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from ureter_semis.demo import (
    _fiber_geometry,
    _fiber_ray,
    _laser_target,
    _preprocess,
    _stone_components,
    _stone_measurement,
)
from ureter_semis.model import build_model


def put_label(img, text, xy, color=(255, 255, 255), scale=0.52):
    """Readable label on a video frame (OpenCV uses BGR)."""
    x, y = int(xy[0]), int(xy[1])
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def segmentation_overlay(frame, pred):
    """Overlay colored semantic predictions without additional text."""
    ph, pw = frame.shape[:2]
    labels = cv2.resize(pred, (pw, ph), interpolation=cv2.INTER_NEAREST)
    out = frame.copy()
    for class_id, color in ((1, (40, 40, 235)), (2, (235, 210, 20))):
        region = labels == class_id
        if region.any():
            out[region] = np.clip(
                out[region].astype(np.float32) * 0.45 + np.array(color) * 0.55,
                0, 255,
            ).astype(np.uint8)
    return out


def annotate_geometry(view, pred):
    """Draw stone / fiber geometry directly in the DISPLAY coordinate system."""
    vh, vw = view.shape[:2]
    ph, pw = pred.shape[:2]
    sx, sy = vw / pw, vh / ph

    def xy(p):
        return int(round(float(p[0]) * sx)), int(round(float(p[1]) * sy))

    stones = _stone_components(pred == 1, min_area=5)
    for stone_id, region in enumerate(stones, 1):
        geom = _stone_measurement(region)
        y, x = np.where(region)
        rect = cv2.minAreaRect(np.column_stack((x, y)).astype(np.float32))
        box = cv2.boxPoints(rect)
        box[:, 0] *= sx
        box[:, 1] *= sy
        cv2.polylines(view, [np.round(box).astype(np.int32)], True,
                      (30, 40, 255), 1, cv2.LINE_AA)
        center = xy((geom["center_x"], geom["center_y"]))
        cv2.drawMarker(view, center, (255, 255, 255), cv2.MARKER_CROSS, 9, 1)
        pos_y = max(52, min(vh - 10, center[1] - 8))
        put_label(view, f"S{stone_id}: {geom['width_px']:.1f}x{geom['length_px']:.1f}px@{pw}",
                  (min(vw - 200, max(6, center[0] + 6)), pos_y), scale=0.42)

    fiber = _fiber_geometry(pred == 2)
    ray = _fiber_ray(fiber["component"]) if fiber is not None else None
    if ray is not None:
        start, tip = xy(ray["proximal"]), xy(ray["tip"])
        cv2.arrowedLine(view, start, tip, (240, 225, 0), 2, cv2.LINE_AA, tipLength=0.12)
        cv2.circle(view, tip, 4, (0, 255, 255), -1)
        _, point, hit = _laser_target(stones, ray)
        if hit is not None and hit["hit"]:
            end = xy(point)
            cv2.line(view, tip, end, (0, 215, 255), 1, cv2.LINE_AA)
            cv2.drawMarker(view, end, (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
            put_label(view, "2D candidate", (max(6, min(vw - 140, end[0] + 5)),
                                              max(52, end[1] - 9)), (0, 255, 0), 0.43)
    info = f"Stones: {len(stones)}   Fiber: {fiber['width_px']:.1f}px@{pw}" if fiber else f"Stones: {len(stones)}   Fiber: none"
    cv2.rectangle(view, (0, 0), (vw, 30), (20, 20, 20), -1)
    put_label(view, info, (8, 21), (230, 230, 230), 0.49)


def assemble(left, right, frame_index, total_frames, video_fps,
             processing_fps, display_fps, model_ms):
    h, w = left.shape[:2]
    header, footer = 44, 44
    canvas = np.full((h + header + footer, 2 * w, 3), 22, dtype=np.uint8)
    canvas[header:header + h, :w] = left
    canvas[header:header + h, w:] = right
    cv2.line(canvas, (w, 0), (w, header + h), (90, 90, 90), 2)
    put_label(canvas, "ORIGINAL VIDEO", (12, 29), (255, 255, 255), 0.76)
    put_label(canvas, "URETERSAM SEGMENTATION + GEOMETRY", (w + 12, 29),
              (120, 240, 200), 0.63)
    progress = f"Frame: {frame_index}/{total_frames}" if total_frames > 0 else f"Frame: {frame_index}"
    put_label(canvas, f"{progress}   Source: {video_fps:.1f} FPS", (12, header + h + 28),
              (240, 240, 240), 0.53)
    put_label(canvas, f"PROC: {processing_fps:5.1f} FPS  LIVE: {display_fps:5.1f} FPS  MODEL: {model_ms:5.1f} ms",
              (w + 10, header + h + 28), (95, 230, 155), 0.48)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="UreterSAM original / processed side-by-side preview")
    parser.add_argument("--source", default="data/videos/video_1.mp4", help="Video path or camera index (e.g., 0)")
    parser.add_argument("--checkpoint", default="runs/endosemis/best.pt")
    parser.add_argument("--output", default="runs/video_test/compare.mp4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--max-speed", action="store_true", help="Do not pace a video file to its source FPS")
    parser.add_argument("--no-save", action="store_true", help="Do not write an output MP4")
    args = parser.parse_args()

    if args.panel_width < 320:
        parser.error("--panel-width must be >= 320")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available. Use --device cpu")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        parser.error(f"Checkpoint does not exist: {ckpt_path}")

    # Only load checkpoints that you have created or otherwise trust.
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    size = int(checkpoint.get("size", 256))
    models = [build_model(pretrained=False).to(device).eval() for _ in range(2)]
    models[0].load_state_dict(checkpoint["model1"])
    models[1].load_state_dict(checkpoint["model2"])
    for model in models:
        model.eval()

    is_camera = args.source.isdecimal()
    cap = cv2.VideoCapture(int(args.source) if is_camera else args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else 0
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(src_fps) or src_fps <= 1:
        src_fps = 25.0
    if src_w <= 0 or src_h <= 0:
        cap.release()
        raise RuntimeError("Invalid source video dimensions")
    view_w = args.panel_width
    view_h = max(1, round(src_h * view_w / src_w))
    out_w, out_h = view_w * 2, view_h + 88

    writer = None
    if not args.no_save:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 src_fps, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot write video to {output_path}")

    process_durations = deque(maxlen=30)
    display_durations = deque(maxlen=30)
    prev_display = None
    frames = 0
    started = time.perf_counter()
    cv2.namedWindow("UreterSAM | Original vs Processed", cv2.WINDOW_NORMAL)
    print(f"Source: {args.source}, {src_w}x{src_h}, {src_fps:.2f} FPS, {total} frames")
    print("Keys: q / Esc to quit. 'PROC FPS' = processing throughput; 'LIVE FPS' = screen refresh rate.")

    try:
        with torch.inference_mode():
            while True:
                t0 = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break
                frames += 1

                tensor = _preprocess(frame, size, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                infer_t0 = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=(device.type == "cuda")):
                    logits = (models[0](tensor) + models[1](tensor)) / 2
                # Moving predicted mask to CPU also synchronizes CUDA.
                pred = logits.argmax(dim=1)[0].byte().cpu().numpy()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                model_ms = (time.perf_counter() - infer_t0) * 1000

                original = cv2.resize(frame, (view_w, view_h), interpolation=cv2.INTER_AREA)
                processed = segmentation_overlay(original, pred)
                annotate_geometry(processed, pred)

                process_durations.append(max(time.perf_counter() - t0, 1e-9))
                proc_fps = len(process_durations) / sum(process_durations)
                live_fps = (len(display_durations) / sum(display_durations)
                            if display_durations else 0.0)
                combined = assemble(original, processed, frames, total, src_fps,
                                    proc_fps, live_fps, model_ms)

                if writer is not None:
                    writer.write(combined)
                cv2.imshow("UreterSAM | Original vs Processed", combined)

                # For recorded files, throttle fast inference to the video's native FPS.
                # If processing is slower than the video, no frames are silently skipped.
                if not is_camera and not args.max_speed:
                    target_time = started + frames / src_fps
                    delay_ms = max(1, int((target_time - time.perf_counter()) * 1000))
                else:
                    delay_ms = 1
                key = cv2.waitKey(delay_ms) & 0xFF
                now = time.perf_counter()
                if prev_display is not None:
                    display_durations.append(max(now - prev_display, 1e-9))
                prev_display = now
                if key in (27, ord("q")):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    print(f"Frames processed: {frames}; wall time: {elapsed:.2f}s; total-average displayed throughput: {frames / elapsed:.2f} FPS")
    if writer is not None:
        print(f"Saved: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
