from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch

from .model import build_model


class SegmentationPredictor:
    """Checkpoint loader used by the fixed two-network G0 front end."""

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
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        self.size = int(state.get("size", 320))
        self.classes = state.get("classes", ["background", "stone", "fiber"])
        if self.classes != ["background", "stone", "fiber"]:
            raise ValueError(f"Unexpected class ordering: {self.classes}")
        self.model_count = int(state.get("model_count", 2))
        if self.model_count not in (1, 2):
            raise ValueError("Unsupported segmentation ensemble size")
        architecture = state.get("architecture", "unet_resnet34")
        if architecture != "unet_resnet34":
            raise ValueError(f"Unsupported frozen architecture: {architecture}")
        self.models = tuple(
            build_model(pretrained=False).to(self.device).eval()
            for _ in range(self.model_count)
        )
        for index, model in enumerate(self.models, 1):
            model.load_state_dict(state[f"model{index}"])
