from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn


VENDORED_SRC = Path(__file__).resolve().parents[1] / "third_party" / "Endo-SemiS" / "src"
if str(VENDORED_SRC) not in sys.path:
    sys.path.insert(0, str(VENDORED_SRC))

from models.unet import Unet  # noqa: E402


def build_model(pretrained: bool = True, classes: int = 3,
                architecture: str = "unet_resnet34", encoder_checkpoint: str | None = None) -> nn.Module:
    if architecture == "segformer_mit_b0":
        import torch
        import segmentation_models_pytorch as smp
        model = smp.Segformer(encoder_name="mit_b0", encoder_weights=None,
                             decoder_segmentation_channels=256, in_channels=3, classes=classes)
        if pretrained:
            if encoder_checkpoint is None:
                raise ValueError("SegFormer pretrained initialization requires a local encoder checkpoint")
            model.encoder.load_state_dict(torch.load(encoder_checkpoint, map_location="cpu", weights_only=True))
        return model
    if architecture != "unet_resnet34" or encoder_checkpoint is not None:
        raise ValueError("Unsupported model architecture or encoder checkpoint")
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


def ensemble_logits(models, images):
    if not models:
        raise ValueError("At least one segmentation model is required")
    logits = models[0](images)
    for model in models[1:]:
        logits = logits + model(images)
    return logits / len(models) if len(models) > 1 else logits
