"""Depth Anything V2 单目深度估计推理封装。

基于 HuggingFace transformers 内置的 depth-anything 模型实现（该实现同时兼容
Depth Anything V1 / V2 的权重），默认加载官方 V2 权重：
    depth-anything/Depth-Anything-V2-Small-hf   (~99M,  最快)
    depth-anything/Depth-Anything-V2-Base-hf    (~390M)
    depth-anything/Depth-Anything-V2-Large-hf   (~1.3G, 最准)

首次运行会从 HuggingFace Hub 自动下载并缓存权重（默认缓存目录 ~/.cache/huggingface）。

用法示例：
    from depth_anything_v2_infer import DepthAnythingV2

    estimator = DepthAnythingV2(model_size="small", device="cuda")
    depth = estimator.infer("frame_0010.png")        # -> np.ndarray, HxW, float32
    estimator.save_visualization(depth, "frame_0010_depth.png")

命令行示例：
    python depth_anything_v2_infer.py --input frame.png --output depth.png
    python depth_anything_v2_infer.py --input images_dir/ --output depth_dir/ --model-size base
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class DepthAnythingV2:
    """Depth Anything V2 推理封装，输出与输入分辨率一致的相对深度图。"""

    def __init__(
        self,
        model_size: str = "small",
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        if model_size not in MODEL_IDS:
            raise ValueError(f"model_size 必须是 {list(MODEL_IDS)} 之一，收到: {model_size}")

        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        model_id = MODEL_IDS[model_size]
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id, dtype=self.dtype)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def infer(self, image: str | Path | Image.Image | np.ndarray) -> np.ndarray:
        """对单张图像做深度推理，返回与原图同分辨率的相对深度图 (H, W) float32。

        数值越大表示越靠近相机（transformers depth-anything 的输出约定）。
        """
        pil_image = self._to_pil(image)

        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device, self.dtype)
        outputs = self.model(**inputs)

        post_processed = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(pil_image.height, pil_image.width)]
        )
        depth = post_processed[0]["predicted_depth"]
        return depth.detach().cpu().float().numpy()

    def infer_batch(self, images: list) -> list[np.ndarray]:
        """依次对多张图像做推理，返回深度图列表（各自保持原分辨率）。"""
        return [self.infer(img) for img in images]

    @property
    def model_id(self) -> str:
        """Return the Hugging Face model identifier used by this estimator."""
        for model_id, candidate in MODEL_IDS.items():
            if candidate == self.model.config._name_or_path:
                return candidate
        return str(self.model.config._name_or_path)

    @staticmethod
    def _to_pil(image: str | Path | Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError(f"不支持的图像类型: {type(image)}")

    @staticmethod
    def save_visualization(depth: np.ndarray, output_path: str | Path) -> None:
        """将深度图归一化并保存为灰度 PNG，方便快速查看。"""
        d_min, d_max = float(depth.min()), float(depth.max())
        normalized = (depth - d_min) / (d_max - d_min + 1e-8)
        gray = (normalized * 255).astype(np.uint8)
        Image.fromarray(gray, mode="L").save(output_path)

    @staticmethod
    def save_raw(depth: np.ndarray, output_path: str | Path) -> None:
        """保存原始深度数值（.npy），用于后续定量处理。"""
        np.save(output_path, depth)


def _collect_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Depth Anything V2 推理脚本")
    parser.add_argument("--input", required=True, help="输入图像文件或目录")
    parser.add_argument("--output", required=True, help="输出深度可视化图（.png）文件或目录")
    parser.add_argument("--model-size", default="small", choices=list(MODEL_IDS), help="模型规模")
    parser.add_argument("--device", default=None, help="cuda / cpu，默认自动选择")
    parser.add_argument("--save-raw", action="store_true", help="额外保存原始深度值 .npy")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    inputs = _collect_inputs(input_path)
    if not inputs:
        raise SystemExit(f"未在 {input_path} 找到图像文件")

    estimator = DepthAnythingV2(model_size=args.model_size, device=args.device)

    is_batch = input_path.is_dir()
    if is_batch:
        output_path.mkdir(parents=True, exist_ok=True)

    for img_path in inputs:
        depth = estimator.infer(img_path)
        if is_batch:
            rel_dir = img_path.relative_to(input_path).parent
            out_dir = output_path / rel_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{img_path.stem}_depth.png"
        else:
            out_file = output_path

        estimator.save_visualization(depth, out_file)
        if args.save_raw:
            estimator.save_raw(depth, out_file.with_suffix(".npy"))
        print(f"{img_path} -> {out_file}")


if __name__ == "__main__":
    main()
