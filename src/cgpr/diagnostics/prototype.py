from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cgpr.diagnostics.feature_bank import FeatureBank


@dataclass
class PrototypeBank:
    prototypes: np.ndarray
    source_counts: list[int]


class PrototypeBuilder:
    def __init__(self, seed_per_class: int):
        self.seed_per_class = seed_per_class

    def build(self, bank: FeatureBank) -> PrototypeBank:
        features = bank.features
        probs = bank.probs
        pseudo = bank.pseudo
        confidence = bank.confidence

        num_classes = probs.shape[1]
        prototypes = []
        source_counts = []

        for class_idx in range(num_classes):
            pseudo_indices = np.where(pseudo == class_idx)[0]

            min_count = max(5, self.seed_per_class // 5)

            if len(pseudo_indices) >= min_count:
                ordered = pseudo_indices[np.argsort(-confidence[pseudo_indices])]
                chosen = ordered[:self.seed_per_class]
                weights = confidence[chosen]
            else:
                ordered = np.argsort(-probs[:, class_idx])
                chosen = ordered[:self.seed_per_class]
                weights = probs[chosen, class_idx]

            weights = np.asarray(weights, dtype=np.float32)
            weights = weights / (weights.sum() + 1e-8)

            prototype = (features[chosen] * weights[:, None]).sum(axis=0)
            prototype = prototype / (np.linalg.norm(prototype) + 1e-8)

            prototypes.append(prototype.astype(np.float32))
            source_counts.append(int(len(chosen)))

        return PrototypeBank(
            prototypes=np.stack(prototypes, axis=0),
            source_counts=source_counts,
        )
