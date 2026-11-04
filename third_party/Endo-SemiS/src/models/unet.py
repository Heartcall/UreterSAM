from typing import Any, Optional, Union, Tuple, Callable
from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading
from .unet_parts import UnetDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(SegmentationModel):
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

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        #self.vit = AutoModelForImageClassification.from_pretrained('facebook/dinov2-small-imagenet1k-1-layer')

        self.name = "u-{}".format(encoder_name)

        self.initialize()


        self.bottleneck_reducer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, return_features: bool = False):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        if not torch.jit.is_tracing() or self.requires_divisible_input_shape:
            self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if return_features:
            features_bottleneck_reduced = self.bottleneck_reducer(features[-1])
            features.append(features_bottleneck_reduced)
            return masks, features
        else:
            return masks




