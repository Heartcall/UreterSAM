from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn


VENDORED_SRC = Path(__file__).resolve().parents[1] / "third_party" / "Endo-SemiS" / "src"
if str(VENDORED_SRC) not in sys.path:
    sys.path.insert(0, str(VENDORED_SRC))

from models.unet import Unet  # noqa: E402


def build_model(pretrained: bool = False, classes: int = 3) -> nn.Module:
    if pretrained:
        raise ValueError("The G0 runtime only loads local frozen checkpoints")
    model = Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=classes,
    )
    for block in model.decoder.blocks:
        block.conv2 = nn.Sequential(block.conv2, nn.Dropout2d(p=0.1))
    return model


def ensemble_logits(models, images):
    if not models:
        raise ValueError("At least one segmentation model is required")
    logits = models[0](images)
    for model in models[1:]:
        logits = logits + model(images)
    return logits / len(models) if len(models) > 1 else logits
