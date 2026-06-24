from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet101_Weights


class ResNet101Features(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.feature_extractor(x).flatten(1)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        output[key] = value

    return output


def extract_model_state(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unsupported checkpoint format. Expected a dictionary.")

    for key in ["model_state", "model_state_dict", "state_dict", "model", "net"]:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return dict(value)

    return dict(checkpoint)


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = extract_model_state(checkpoint)
    state = strip_module_prefix(state)

    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    if missing:
        print("First missing keys:", missing[:5])
    if unexpected:
        print("First unexpected keys:", unexpected[:5])

    if len(missing) > 5 or len(unexpected) > 5:
        raise RuntimeError(
            "Checkpoint was not loaded correctly. Too many missing/unexpected keys."
        )


class SourceModelLoader:
    def __init__(self, num_classes: int, checkpoint_path: Path, device: torch.device):
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        self.device = device

    def build(self) -> ResNet101Features:
        model = ResNet101Features(num_classes=self.num_classes).to(self.device)
        load_checkpoint(model, self.checkpoint_path, self.device)
        model.eval()
        return model
