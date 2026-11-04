"""Shared, stateless A0 inference. No temporal or geometric post-processing."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch

from .data import MEAN, STD
from .model import build_model, ensemble_logits


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(frame: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    # Keep the interpolation and normalization identical to the original demo.
    rgb = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB)
    array = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy((array - MEAN) / STD).unsqueeze(0).to(device)


@dataclass
class Prediction:
    mask: np.ndarray
    probabilities: np.ndarray
    preprocess_ms: float
    model_ms: float
    postprocess_ms: float


class SegmentationPredictor:
    def __init__(self, checkpoint: str | Path, device: str = "cuda:0", seed: int = 42):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; explicitly select --device cpu")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.checkpoint_path = Path(checkpoint).resolve()
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.size = int(state.get("size", 320))
        self.classes = state.get("classes", ["background", "stone", "fiber"])
        if self.classes != ["background", "stone", "fiber"]:
            raise ValueError(f"Unexpected class ordering: {self.classes}")
        self.model_count = int(state.get("model_count", 2))
        if self.model_count not in (1, 2):
            raise ValueError("Unsupported segmentation ensemble size")
        self.architecture = state.get("architecture", "unet_resnet34")
        model_args = {} if self.architecture == "unet_resnet34" else {"architecture": self.architecture}
        self.models = tuple(build_model(pretrained=False, **model_args).to(self.device).eval() for _ in range(self.model_count))
        for index, model in enumerate(self.models, 1):
            model.load_state_dict(state[f"model{index}"])

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict(self, frame: np.ndarray) -> Prediction:
        self._sync()
        start = time.perf_counter()
        tensor = preprocess(frame, self.size, self.device)
        self._sync()
        prepared = time.perf_counter()
        logits = ensemble_logits(self.models, tensor)
        self._sync()
        inferred = time.perf_counter()
        probabilities = torch.softmax(logits, dim=1)
        # Argmax in float32 before optional storage quantization.
        mask = probabilities.argmax(dim=1)[0].byte().cpu().numpy()
        probability_array = probabilities[0].cpu().numpy()
        self._sync()
        end = time.perf_counter()
        return Prediction(mask, probability_array, (prepared-start)*1000,
                          (inferred-prepared)*1000, (end-inferred)*1000)
