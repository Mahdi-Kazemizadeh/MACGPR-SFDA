from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class FeatureBank:
    features: np.ndarray
    logits: np.ndarray
    probs: np.ndarray
    confidence: np.ndarray
    pseudo: np.ndarray
    entropy: np.ndarray
    true_labels: np.ndarray


class TargetFeatureExtractor:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def extract(self, loader: DataLoader) -> FeatureBank:
        self.model.eval()

        all_logits = []
        all_features = []
        all_labels = []

        for step, (images, labels) in enumerate(loader, 1):
            images = images.to(self.device, non_blocking=True)
            logits, features = self.model(images, return_features=True)

            all_logits.append(logits.detach().cpu())
            all_features.append(features.detach().cpu())
            all_labels.append(labels.detach().cpu())

            if step % 50 == 0:
                print(f"Extracted batches: {step}/{len(loader)}")

        logits_t = torch.cat(all_logits, dim=0).float()
        features_t = torch.cat(all_features, dim=0).float()
        labels_t = torch.cat(all_labels, dim=0).long()

        probs_t = F.softmax(logits_t, dim=1)
        confidence_t, pseudo_t = probs_t.max(dim=1)
        entropy_t = -(probs_t * torch.log(probs_t.clamp_min(1e-8))).sum(dim=1)

        features_t = F.normalize(features_t, dim=1)

        return FeatureBank(
            features=features_t.numpy(),
            logits=logits_t.numpy(),
            probs=probs_t.numpy(),
            confidence=confidence_t.numpy(),
            pseudo=pseudo_t.numpy(),
            entropy=entropy_t.numpy(),
            true_labels=labels_t.numpy(),
        )
