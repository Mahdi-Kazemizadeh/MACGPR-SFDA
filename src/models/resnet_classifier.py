from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet101_Weights


class ResNetClassifier(nn.Module):
    """ResNet-101 classifier that can return logits and feature vectors."""

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet101",
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if backbone != "resnet101":
            raise ValueError(f"Unsupported backbone: {backbone}")

        weights = ResNet101_Weights.DEFAULT if pretrained else None
        base_model = models.resnet101(weights=weights)

        self.feature_extractor = nn.Sequential(
            *list(base_model.children())[:-1])
        self.classifier = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x).flatten(1)
        logits = self.classifier(features)

        if return_features:
            return logits, features

        return logits

    def freeze_classifier(self) -> None:
        """Freeze the classifier head."""
        for parameter in self.classifier.parameters():
            parameter.requires_grad = False

    def unfreeze_classifier(self) -> None:
        """Unfreeze the classifier head."""
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

    def freeze_feature_extractor(self) -> None:
        """Freeze the feature extractor."""
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

    def unfreeze_feature_extractor(self) -> None:
        """Unfreeze the feature extractor."""
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = True

    def count_trainable_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
