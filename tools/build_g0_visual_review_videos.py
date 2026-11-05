"""Build high-resolution source/Raw2/G0 videos for human visual review.

The script re-runs the frozen two-network model, the accepted S2 input path,
and the latest no-fallback G0 tapered tangent-cap implementation. Historical
source and S2 hashes are checked before encoding, while G0 is intentionally
generated from the current implementation and no-fallback configuration.
"""
from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ureter_semis.experiment import atomic_json, seed_training
from .fiber_photometric_input import infer
from .fiber_reference_workspace import ROOT, read, sha256
from .fiber_tapered_tangent_cap import TaperedTangentCap
from .fiber_weak_stage2 import GrowthPipeline
from .replay_classwise_experts import model_state_hash
from .run_fiber_temporal_state_control import construct
from .run_two_network_experts import array_hash


NETWORK_PREFLIGHT = ROOT / "runs/a1_weak_blue_evidence_20261103/preflight.json"
SOURCE_RUN = ROOT / "runs/a1_weak_blue_evidence_20261103"
STAGE_CONFIG = ROOT / "configs/weak_blue_evidence_control_20261103.json"
STAGE_T1_CONFIG = ROOT / "configs/fiber_temporal_observation_search_20261030.json"
G0_CONFIG = ROOT / "configs/tapered_tangent_cap_no_fallback_20261104.json"
G0_HISTORY = ROOT / "runs/a1_tapered_tangent_cap_20261104/comparison"
TIMING_HELPER = ROOT / "runs/engineering/timing_bisector_media_20261103.py"
FFMPEG_IDENTITY = ROOT / "runs/dependencies/ffmpeg_media_20261103/identity.json"
DEFAULT_OUTPUT = ROOT / "runs/g0_visual_review_no_fallback_20261105"

PANEL = 640
HEADER = 86
FOOTER = 94
CANVAS_SIZE = (PANEL * 3, HEADER + PANEL + FOOTER)
STONE_BGR = np.array([38, 45, 232], np.float32)
FIBER_BGR = np.array([235, 211, 24], np.float32)
ADD_BGR = (47, 220, 85)
REMOVE_BGR = (0, 145, 255)

REASON_CN = {
    "converging_geometry_reconstruction": "几何重建已应用",
    "converging_geometry_reconstruction_no_fallback": "无回退几何已应用",
    "geometric_stone_conflict": "与结石冲突",
    "tip_at_or_beyond_side_intersection": "尖端达到或越过边线交点",
    "subject_not_slender": "主体细长度不足",
    "width_out_of_range": "宽度超出范围",
    "no_trusted_subject": "没有唯一可信主体",
    "ambiguous_trusted_subjects": "可信主体存在歧义",
    "insufficient_fitted_span": "有效拟合跨度不足",
    "parallel_or_ill_conditioned_line_intersection": "边线交点不稳定",
    "no_blue_external_terminal": "未找到外侧蓝色尖端证据",
}


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(24)
FONT_PANEL = load_font(25)
FONT_SMALL = load_font(20)


def centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str,
             font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = box[0] + (box[2] - box[0] - width) / 2
    y = box[1] + (box[3] - box[1] - height) / 2 - bounds[1]
    draw.text((x, y), text, font=font, fill=fill)


def overlay_mask(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.38) -> np.ndarray:
    result = frame.copy()
    for label, color in [(1, STONE_BGR), (2, FIBER_BGR)]:
        selected = mask == label
        result[selected] = np.clip(
            result[selected].astype(np.float32) * (1 - alpha) + color * alpha, 0, 255
        ).astype(np.uint8)
        contours, _ = cv2.findContours(selected.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, tuple(int(x) for x in color), 1,
                         lineType=cv2.LINE_AA)
    return result


def panel_image(frame: np.ndarray, mask: np.ndarray | None = None,
                added: np.ndarray | None = None,
                removed: np.ndarray | None = None) -> np.ndarray:
    image = frame if mask is None else overlay_mask(frame, mask)
    image = cv2.resize(image, (PANEL, PANEL), interpolation=cv2.INTER_LANCZOS4)
    for edit, color in [(removed, REMOVE_BGR), (added, ADD_BGR)]:
        if edit is None or not np.any(edit):
            continue
        large = cv2.resize(edit.astype(np.uint8), (PANEL, PANEL),
                           interpolation=cv2.INTER_NEAREST)
        selected = large > 0
        color_array = np.asarray(color, np.float32)
        image[selected] = np.clip(
            image[selected].astype(np.float32) * 0.30 + color_array * 0.70, 0, 255
        ).astype(np.uint8)
    return image


def legend_item(draw: ImageDraw.ImageDraw, x: int, y: int,
                color: tuple[int, int, int], text: str) -> int:
    draw.rounded_rectangle((x, y + 2, x + 24, y + 26), radius=4, fill=color)
    draw.text((x + 32, y), text, font=FONT_SMALL, fill=(224, 230, 238))
    bounds = draw.textbbox((0, 0), text, font=FONT_SMALL)
    return x + 32 + bounds[2] - bounds[0] + 28


def render(frame: np.ndarray, raw: np.ndarray, g0_mask: np.ndarray,
           g0_info: dict, added: np.ndarray, removed: np.ndarray,
           video_id: str, index: int, total: int, pts: float,
           source_fps: float, raw2_fps: float, g0_fps: float) -> np.ndarray:
    panels = [
        panel_image(frame),
        panel_image(frame, raw),
        panel_image(frame, g0_mask, added, removed),
    ]
    canvas = np.full((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), (23, 27, 35), np.uint8)
    canvas[HEADER:HEADER + PANEL] = np.hstack(panels)
    for x in [PANEL, PANEL * 2]:
        cv2.line(canvas, (x, HEADER), (x, HEADER + PANEL), (78, 84, 96), 2)

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    title = (
        f"G0 人工视觉评审  |  {video_id}  |  帧 {index + 1:04d}/{total:04d}"
        f"（索引 {index:04d}）  |  PTS {pts:8.3f} s"
    )
    centered(draw, (0, 0, CANVAS_SIZE[0], 38), title, FONT_TITLE, (239, 243, 249))
    labels = ["原始视频（无叠加）", "两网初始分割（Raw2）", "G0 几何结果"]
    for column, label in enumerate(labels):
        centered(draw, (column * PANEL, 38, (column + 1) * PANEL, HEADER),
                 label, FONT_PANEL, (239, 243, 249))

    applied = bool(g0_info["applied"])
    reason = REASON_CN.get(g0_info["reason"], g0_info["reason"])
    status = "G0 已应用（无提案回退）" if applied else f"G0 不可构造：{reason}"
    badge_color = (29, 144, 82) if applied else (143, 81, 54)
    badge_box = (PANEL * 2 + 14, HEADER + 14, PANEL * 3 - 14, HEADER + 54)
    draw.rounded_rectangle(badge_box, radius=10, fill=badge_color)
    centered(draw, badge_box, status, FONT_SMALL, (255, 255, 255))

    footer_y = HEADER + PANEL + 10
    x = 24
    x = legend_item(draw, x, footer_y, (232, 45, 38), "结石")
    x = legend_item(draw, x, footer_y, (24, 211, 235), "光纤")
    x = legend_item(draw, x, footer_y, (85, 220, 47), "G0 新增区域（相对 Raw2）")
    x = legend_item(draw, x, footer_y, (255, 145, 0), "G0 删除区域（相对 Raw2）")
    delta = f"本帧变化：+{int(added.sum())} / -{int(removed.sum())} px"
    draw.text((CANVAS_SIZE[0] - 370, footer_y), delta, font=FONT_SMALL,
              fill=(224, 230, 238))
    fps_texts = [
        f"源视频实时帧率  {source_fps:5.2f} FPS",
        f"两网 Raw2 实测帧率  {raw2_fps:5.2f} FPS",
        f"G0 总流程实测帧率  {g0_fps:5.2f} FPS",
    ]
    for column, text in enumerate(fps_texts):
        centered(draw, (column * PANEL, footer_y + 35, (column + 1) * PANEL,
                         CANVAS_SIZE[1]), text, FONT_SMALL, (181, 218, 255))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


class Encoder:
    def __init__(self, ffmpeg: Path, path: Path, fps: float, crf: int):
        self.path = path
        command = [
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s:v", f"{CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}",
            "-r", f"{fps:.12f}", "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            "-bf", "0", "-pix_fmt", "yuv420p", str(path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                        stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (CANVAS_SIZE[1], CANVAS_SIZE[0], 3):
            raise ValueError(f"Unexpected review frame shape: {frame.shape}")
        assert self.process.stdin is not None
        self.process.stdin.write(frame.tobytes())

    def close(self) -> None:
        assert self.process.stdin is not None and self.process.stderr is not None
        self.process.stdin.close()
        error = self.process.stderr.read().decode("utf-8", errors="replace")
        code = self.process.wait()
        if code:
            raise RuntimeError(f"ffmpeg failed ({code}): {error}")


def timing_module():
    spec = importlib.util.spec_from_file_location("g0_source_timing", TIMING_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load timing helper: {TIMING_HELPER}")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def output_name(video_id: str) -> str:
    return f"{video_id}_original_raw2_g0_latest_review_1920x820.mp4"


def process_video(video_id: str, output: Path, crf: int, focus: tuple[int, int] | None,
                  poster_display_frame: int | None) -> dict:
    preflight = read(NETWORK_PREFLIGHT)
    video = next(item for item in preflight["videos"] if item["video_id"] == video_id)
    if sha256(video["path"]) != video["sha256"]:
        raise ValueError(f"Source video changed: {video_id}")
    for checkpoint in preflight["checkpoints"].values():
        if sha256(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError(f"Checkpoint changed: {checkpoint['path']}")

    source = read(SOURCE_RUN / f"source_{video_id}.json")["frames"]
    historical = read(G0_HISTORY / f"{video_id}_sync.json")
    if len(source) != video["frames"] or len(historical) != video["frames"]:
        raise ValueError(f"Historical frame count mismatch: {video_id}")
    stage_cfg = read(STAGE_CONFIG)
    t1_cfg = read(STAGE_T1_CONFIG)
    g0_cfg = read(G0_CONFIG)
    media_timing = read(SOURCE_RUN / "source_media_timing.json")[video_id]
    fps = float(media_timing["average_fps"])
    ffmpeg = Path(read(FFMPEG_IDENTITY)["binary"])
    if not ffmpeg.exists():
        raise FileNotFoundError(ffmpeg)

    seed_training(42)
    experts = construct(preflight)
    model_before = model_state_hash(experts)
    stage2 = GrowthPipeline(stage_cfg, "weak_blue_extension_tip", t1_cfg, True)
    g0 = TaperedTangentCap(g0_cfg)

    cap = cv2.VideoCapture(str(video["path"]))
    ok, first = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot decode first frame: {video_id}")
    for warmup_index in range(12):
        raw, _, fiber_probability, _ = infer(experts, first, "raw2")
        stage2.run(first, raw, fiber_probability, warmup_index / video["fps"], video_id)
    stage2.reset()

    draft = output / (Path(output_name(video_id)).stem + "_draft.mp4")
    encoder = Encoder(ffmpeg, draft, fps, crf)
    focus_encoder = None
    focus_draft = None
    if focus is not None:
        focus_draft = output / "video_2_frames1233-1291_original_raw2_g0_latest_review_1920x820_draft.mp4"
        focus_encoder = Encoder(ffmpeg, focus_draft, fps, crf)

    cap = cv2.VideoCapture(str(video["path"]))
    frame_index = 0
    applied = 0
    fallback = 0
    added_total = 0
    removed_total = 0
    raw2_ms_window: deque[float] = deque(maxlen=30)
    g0_ms_window: deque[float] = deque(maxlen=30)
    started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            record = source[frame_index]
            rgb_hash = array_hash(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if rgb_hash != record["rgb_raw_sha256"]:
                raise ValueError(f"Source RGB mismatch: {video_id} frame {frame_index}")
            pipeline_start = time.perf_counter()
            raw, _, fiber_probability, network_timing = infer(experts, frame, "raw2")
            raw2_ms_window.append(float(network_timing["pipeline_total_ms"]))
            stage2.run(frame, raw, fiber_probability, record["timestamp_seconds"], video_id)
            old = stage2.final_mask.copy()
            g0.run(frame, raw, fiber_probability, old)
            g0_ms_window.append((time.perf_counter() - pipeline_start) * 1000)
            history = historical[frame_index]
            if array_hash(old) != history["old_mask_sha256"]:
                raise ValueError(f"S2 parity mismatch: {video_id} frame {frame_index}")

            # The visual-review delta is defined against the two-network Raw2
            # result, not against the S2 mask used internally as G0's input.
            added = (g0.final_mask == 2) & (raw != 2)
            removed = (raw == 2) & (g0.final_mask != 2)
            duration = media_timing["durations"][frame_index]
            source_fps = media_timing["timescale"] / duration if duration else 0.0
            raw2_fps = 1000.0 / (sum(raw2_ms_window) / len(raw2_ms_window))
            total_fps = 1000.0 / (sum(g0_ms_window) / len(g0_ms_window))
            review = render(frame, raw, g0.final_mask, g0.info, added, removed,
                            video_id, frame_index, video["frames"],
                            record["timestamp_seconds"], source_fps, raw2_fps, total_fps)
            encoder.write(review)
            if focus_encoder is not None and focus is not None and focus[0] <= frame_index < focus[1]:
                focus_encoder.write(review)
            if poster_display_frame is not None and frame_index + 1 == poster_display_frame:
                cv2.imwrite(str(output / f"{video_id}_frame{poster_display_frame:04d}_poster.png"),
                            review, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            applied += int(g0.info["applied"])
            fallback += int(not g0.info["applied"])
            added_total += int(added.sum())
            removed_total += int(removed.sum())
            frame_index += 1
            if frame_index % 200 == 0:
                print(json.dumps({"video": video_id, "frames": frame_index,
                                  "total": video["frames"]}, ensure_ascii=False), flush=True)
    finally:
        cap.release()
        encoder.close()
        if focus_encoder is not None:
            focus_encoder.close()

    if frame_index != video["frames"]:
        raise ValueError(f"Decoded {frame_index}, expected {video['frames']}: {video_id}")
    if model_state_hash(experts) != model_before:
        raise ValueError("Model state changed during review rendering")
    stage2.reset()
    del experts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    timing = timing_module()
    final = output / output_name(video_id)
    timed = timing.retime(draft, final, media_timing)
    draft.unlink()
    focus_result = None
    if focus_draft is not None and focus is not None:
        focus_final = output / "video_2_frames1233-1291_original_raw2_g0_latest_review_1920x820.mp4"
        focus_result = timing.retime(focus_draft, focus_final, media_timing,
                                    focus[0], focus[1])
        focus_draft.unlink()
    return {
        "video_id": video_id,
        "source": video["path"],
        "source_sha256": video["sha256"],
        "frames": frame_index,
        "resolution": list(CANVAS_SIZE),
        "applied": applied,
        "fallback": fallback,
        "added_pixels": added_total,
        "removed_pixels": removed_total,
        "historical_source_and_s2_hashes_all_match": True,
        "g0_config": str(G0_CONFIG),
        "g0_config_sha256": sha256(G0_CONFIG),
        "g0_source_sha256": sha256(ROOT / "tools/fiber_tapered_tangent_cap.py"),
        "source_pts_exact": timed["all_packet_PTS_match_source"],
        "encoded_sample_bytes_unchanged_by_retime": timed["all_encoded_sample_bytes_unchanged"],
        "output": str(final),
        "output_sha256": sha256(final),
        "output_bytes": final.stat().st_size,
        "focus_output": focus_result["path"] if focus_result else None,
        "focus_sha256": focus_result["sha256"] if focus_result else None,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--crf", type=int, default=14)
    parser.add_argument("--videos", nargs="+", choices=["video_1", "video_2"],
                        default=["video_1", "video_2"])
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    results = []
    for video_id in args.videos:
        focus = (1232, 1291) if video_id == "video_2" else None
        poster = 1250 if video_id == "video_2" else 317
        results.append(process_video(video_id, output, args.crf, focus, poster))
    manifest = {
        "purpose": "human_visual_review_original_vs_two_network_raw2_vs_G0",
        "g0_definition": "latest no-fallback converging sides + RGB axis vertex T + tangent circle; constructible geometry always applies and raw stone overlap is clipped",
        "layout": ["original_unmarked", "two_network_initial_Raw2", "G0_actual_result"],
        "delta_baseline": "two-network initial segmentation Raw2",
        "delta_definition": {
            "G0_add_area": "G0 fiber minus Raw2 fiber",
            "G0_remove_area": "Raw2 fiber minus G0 fiber",
        },
        "colors": {"stone": "red", "fiber": "cyan", "G0_add_area": "green",
                   "G0_remove_area": "orange"},
        "live_comparison": "source FPS vs rolling-30-frame measured Raw2 FPS vs rolling-30-frame measured network+S2+G0 FPS",
        "alpha": 0.38,
        "crf": args.crf,
        "font": "Microsoft YaHei with local fallbacks",
        "videos": results,
        "physical_tip_truth": None,
        "clinical_claim": False,
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
