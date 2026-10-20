from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn


VENDORED_SRC = Path(__file__).resolve().parents[1] / "third_party" / "Endo-SemiS" / "src"
if str(VENDORED_SRC) not in sys.path:
    sys.path.insert(0, str(VENDORED_SRC))

from models.unet import Unet  # noqa: E402


def build_model(pretrained: bool = True, classes: int = 3) -> nn.Module:
    model = Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=classes,
    )
    for block in model.decoder.blocks:
        block.conv2 = nn.Sequential(block.conv2, nn.Dropout2d(p=0.1))
    return model


def enable_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()
