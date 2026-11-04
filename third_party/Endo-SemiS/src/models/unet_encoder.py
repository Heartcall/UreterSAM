from typing import Any, Optional, Union, Tuple, Callable
from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading
import torch
import torch.nn as nn


class Unet_encoder(SegmentationModel):
    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable]] = None,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        #self.vit = AutoModelForImageClassification.from_pretrained('facebook/dinov2-small-imagenet1k-1-layer')

        self.name = "u-{}".format(encoder_name)

        self.bottleneck_reducer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )


    def forward(self, x):
        if not torch.jit.is_tracing() or self.requires_divisible_input_shape:
            self.check_input_shape(x)

        features = self.encoder(x)
        features_bottleneck_reduced = self.bottleneck_reducer(features[-1])
        features.append(features_bottleneck_reduced)
        return features




