from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet101_Weights,
    ConvNeXt_Base_Weights,
    ViT_B_16_Weights,
    Swin_B_Weights,
)


class BackboneClassifier(nn.Module):
    """
    Backbone-aware image classifier for source training and source-only evaluation.

    Supported backbones:
        - resnet101
        - convnext_base
        - vit_b_16
        - swin_b

    This module is intended for backbone comparison first.
    Feature extraction for CGPR adaptation can be added after we identify
    which backbone is worth using.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes

        if self.backbone_name == "resnet101":
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            model = models.resnet101(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

        elif self.backbone_name == "convnext_base":
            weights = ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            model = models.convnext_base(weights=weights)
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, num_classes)

        elif self.backbone_name == "vit_b_16":
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            model = models.vit_b_16(weights=weights)
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)

        elif self.backbone_name == "swin_b":
            weights = Swin_B_Weights.DEFAULT if pretrained else None
            model = models.swin_b(weights=weights)
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)

        else:
            raise ValueError(
                f"Unsupported backbone: {backbone_name}. "
                "Choose one of: resnet101, convnext_base, vit_b_16, swin_b"
            )

        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_backbone_classifier(
    backbone_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> BackboneClassifier:
    return BackboneClassifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )
