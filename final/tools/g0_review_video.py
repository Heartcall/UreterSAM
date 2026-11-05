"""Standalone Raw2/G0 comparison-video rendering for the formal G0 runner."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANEL = 640
HEADER = 86
FOOTER = 94
CANVAS_SIZE = (PANEL * 3, HEADER + PANEL + FOOTER)
STONE_BGR = np.array([38, 45, 232], np.float32)
FIBER_BGR = np.array([235, 211, 24], np.float32)
ADD_BGR = (47, 220, 85)
REMOVE_BGR = (0, 145, 255)

REASON_CN = {
    "converging_geometry_reconstruction_no_fallback": "无回退几何已应用",
    "no_trusted_subject": "没有唯一可信主体",
    "ambiguous_trusted_subjects": "可信主体存在歧义",
    "old_subject_not_unique": "内部主体不唯一",
    "insufficient_visible_side_pairs": "可见边线点不足",
    "parallel_or_ill_conditioned_line_intersection": "边线交点不稳定",
    "sides_diverge_in_distal_direction": "边线远端不收敛",
    "no_blue_external_terminal": "未找到外侧蓝色尖端证据",
    "tip_at_or_beyond_side_intersection": "尖端达到或越过边线交点",
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = _font(24)
FONT_PANEL = _font(25)
FONT_SMALL = _font(20)


def _centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
              text: str, font: ImageFont.ImageFont,
              fill: tuple[int, int, int]) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = box[0] + (box[2] - box[0] - width) / 2
    y = box[1] + (box[3] - box[1] - height) / 2 - bounds[1]
    draw.text((x, y), text, font=font, fill=fill)


def _overlay_mask(frame: np.ndarray, mask: np.ndarray,
                  alpha: float = 0.38) -> np.ndarray:
    result = frame.copy()
    for label, color in [(1, STONE_BGR), (2, FIBER_BGR)]:
        selected = mask == label
        result[selected] = np.clip(
            result[selected].astype(np.float32) * (1 - alpha) + color * alpha,
            0, 255,
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            result, contours, -1, tuple(int(value) for value in color), 1,
            lineType=cv2.LINE_AA,
        )
    return result


def _panel(frame: np.ndarray, mask: np.ndarray | None = None,
           added: np.ndarray | None = None,
           removed: np.ndarray | None = None) -> np.ndarray:
    image = frame if mask is None else _overlay_mask(frame, mask)
    image = cv2.resize(image, (PANEL, PANEL), interpolation=cv2.INTER_LANCZOS4)
    for edit, color in [(removed, REMOVE_BGR), (added, ADD_BGR)]:
        if edit is None or not np.any(edit):
            continue
        selected = cv2.resize(
            edit.astype(np.uint8), (PANEL, PANEL), interpolation=cv2.INTER_NEAREST
        ) > 0
        color_array = np.asarray(color, np.float32)
        image[selected] = np.clip(
            image[selected].astype(np.float32) * 0.30 + color_array * 0.70,
            0, 255,
        ).astype(np.uint8)
    return image


def _legend(draw: ImageDraw.ImageDraw, x: int, y: int,
            color: tuple[int, int, int], text: str) -> int:
    draw.rounded_rectangle((x, y + 2, x + 24, y + 26), radius=4, fill=color)
    draw.text((x + 32, y), text, font=FONT_SMALL, fill=(224, 230, 238))
    bounds = draw.textbbox((0, 0), text, font=FONT_SMALL)
    return x + 32 + bounds[2] - bounds[0] + 28


def render_review(frame: np.ndarray, raw: np.ndarray, g0_mask: np.ndarray,
                  g0_info: dict, added: np.ndarray, removed: np.ndarray,
                  video_id: str, index: int, total: int, pts: float,
                  source_fps: float, raw2_fps: float,
                  g0_fps: float) -> np.ndarray:
    panels = [
        _panel(frame),
        _panel(frame, raw),
        _panel(frame, g0_mask, added, removed),
    ]
    canvas = np.full((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), (23, 27, 35), np.uint8)
    canvas[HEADER:HEADER + PANEL] = np.hstack(panels)
    for x in [PANEL, PANEL * 2]:
        cv2.line(canvas, (x, HEADER), (x, HEADER + PANEL), (78, 84, 96), 2)

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    total_text = f"{total:04d}" if total > 0 else "????"
    title = (
        f"G0 人工视觉评审  |  {video_id}  |  帧 {index + 1:04d}/{total_text}"
        f"（索引 {index:04d}）  |  时间 {pts:8.3f} s"
    )
    _centered(draw, (0, 0, CANVAS_SIZE[0], 38), title, FONT_TITLE,
              (239, 243, 249))
    labels = ["原始视频（无叠加）", "两网初始分割（Raw2）", "G0 几何结果"]
    for column, label in enumerate(labels):
        _centered(draw, (column * PANEL, 38, (column + 1) * PANEL, HEADER),
                  label, FONT_PANEL, (239, 243, 249))

    applied = bool(g0_info["applied"])
    reason = REASON_CN.get(g0_info["reason"], g0_info["reason"])
    status = "G0 已应用（无提案回退）" if applied else f"G0 不可构造：{reason}"
    badge = (PANEL * 2 + 14, HEADER + 14, PANEL * 3 - 14, HEADER + 54)
    draw.rounded_rectangle(badge, radius=10,
                           fill=(29, 144, 82) if applied else (143, 81, 54))
    _centered(draw, badge, status, FONT_SMALL, (255, 255, 255))

    footer_y = HEADER + PANEL + 10
    x = 24
    x = _legend(draw, x, footer_y, (232, 45, 38), "结石")
    x = _legend(draw, x, footer_y, (24, 211, 235), "光纤")
    x = _legend(draw, x, footer_y, (85, 220, 47), "G0 新增区域（相对 Raw2）")
    _legend(draw, x, footer_y, (255, 145, 0), "G0 删除区域（相对 Raw2）")
    delta = f"本帧变化：+{int(added.sum())} / -{int(removed.sum())} px"
    draw.text((CANVAS_SIZE[0] - 370, footer_y), delta, font=FONT_SMALL,
              fill=(224, 230, 238))
    fps_texts = [
        f"源视频帧率  {source_fps:5.2f} FPS",
        f"两网 Raw2 实测帧率  {raw2_fps:5.2f} FPS",
        f"G0 总流程实测帧率  {g0_fps:5.2f} FPS",
    ]
    for column, text in enumerate(fps_texts):
        _centered(
            draw,
            (column * PANEL, footer_y + 35, (column + 1) * PANEL, CANVAS_SIZE[1]),
            text, FONT_SMALL, (181, 218, 255),
        )
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError(
            "comparison video requires imageio-ffmpeg; install final/requirements.txt"
        ) from error
    return imageio_ffmpeg.get_ffmpeg_exe()


class ReviewVideoWriter:
    def __init__(self, path: Path, fps: float, total: int,
                 video_id: str, crf: int = 14):
        self.path = path
        self.fps = fps
        self.total = total
        self.video_id = video_id
        self.crf = crf
        self.raw2_window: deque[float] = deque(maxlen=30)
        self.g0_window: deque[float] = deque(maxlen=30)
        self.frames = 0
        self.added_pixels = 0
        self.removed_pixels = 0
        command = [
            _ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s:v", f"{CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}",
            "-r", f"{fps:.12f}", "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            "-bf", "0", "-pix_fmt", "yuv420p", str(path),
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.closed = False

    def write(self, frame: np.ndarray, raw: np.ndarray, g0_mask: np.ndarray,
              g0_info: dict, index: int, pts: float,
              network_ms: float, g0_total_ms: float) -> dict:
        self.raw2_window.append(network_ms)
        self.g0_window.append(g0_total_ms)
        raw2_fps = 1000.0 / (sum(self.raw2_window) / len(self.raw2_window))
        g0_fps = 1000.0 / (sum(self.g0_window) / len(self.g0_window))
        added = (g0_mask == 2) & (raw != 2)
        removed = (raw == 2) & (g0_mask != 2)
        review = render_review(
            frame, raw, g0_mask, g0_info, added, removed,
            self.video_id, index, self.total, pts, self.fps, raw2_fps, g0_fps,
        )
        if review.shape != (CANVAS_SIZE[1], CANVAS_SIZE[0], 3):
            raise ValueError(f"Unexpected review frame shape: {review.shape}")
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg input pipe is closed")
        self.process.stdin.write(review.tobytes())
        added_count = int(added.sum())
        removed_count = int(removed.sum())
        self.frames += 1
        self.added_pixels += added_count
        self.removed_pixels += removed_count
        return {
            "added_pixels_vs_raw2": added_count,
            "removed_pixels_vs_raw2": removed_count,
            "raw2_rolling_30_fps": raw2_fps,
            "g0_rolling_30_fps": g0_fps,
        }

    def close(self) -> None:
        if self.closed:
            return
        if self.process.stdin is None or self.process.stderr is None:
            raise RuntimeError("FFmpeg pipes are unavailable")
        self.process.stdin.close()
        error = self.process.stderr.read().decode("utf-8", errors="replace")
        code = self.process.wait()
        if code:
            raise RuntimeError(f"FFmpeg failed ({code}): {error}")
        if not self.path.is_file() or self.path.stat().st_size == 0:
            raise RuntimeError(f"Comparison video was not created: {self.path}")
        self.closed = True

    def abort(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.path.unlink(missing_ok=True)

    def summary(self) -> dict:
        return {
            "file": self.path.name,
            "frames": self.frames,
            "resolution": list(CANVAS_SIZE),
            "codec": "H.264/libx264",
            "crf": self.crf,
            "timing": "constant frame rate from source-reported average FPS",
            "delta_baseline": "two-network initial segmentation Raw2",
            "added_pixels_vs_raw2": self.added_pixels,
            "removed_pixels_vs_raw2": self.removed_pixels,
        }
