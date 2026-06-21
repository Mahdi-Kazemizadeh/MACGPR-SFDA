from typing import Any

import numpy as np


class ReliabilityEstimator:
    """Computes target-label-free reliability scores for pseudo-label selection."""

    def __init__(self, num_classes: int, config: dict[str, Any]) -> None:
        self.num_classes = num_classes
        self.config = config

    def compute(
        self,
        probabilities: np.ndarray,
        features: np.ndarray,
        pseudo_labels: np.ndarray,
        clusters: np.ndarray,
        cluster_refined_labels: np.ndarray,
    ) -> np.ndarray:
        confidence = probabilities.max(axis=1)
        entropy_score = self._entropy_score(probabilities)
        cluster_agreement = self._cluster_agreement(
            pseudo_labels=pseudo_labels,
            cluster_refined_labels=cluster_refined_labels,
        )
        prototype_similarity = self._prototype_similarity(
            features=features,
            pseudo_labels=cluster_refined_labels,
        )

        weights = self.config["adaptation"].get("reliability", {})

        confidence_weight = float(weights.get("confidence_weight", 0.40))
        entropy_weight = float(weights.get("entropy_weight", 0.20))
        cluster_weight = float(weights.get("cluster_weight", 0.25))
        prototype_weight = float(weights.get("prototype_weight", 0.15))

        reliability = (
            confidence_weight * confidence
            + entropy_weight * entropy_score
            + cluster_weight * cluster_agreement
            + prototype_weight * prototype_similarity
        )

        return np.clip(reliability, 0.0, 1.0)

    def _entropy_score(self, probabilities: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        entropy = -np.sum(probabilities * np.log(probabilities + eps), axis=1)
        normalized_entropy = entropy / np.log(self.num_classes)
        return 1.0 - normalized_entropy

    @staticmethod
    def _cluster_agreement(
        pseudo_labels: np.ndarray,
        cluster_refined_labels: np.ndarray,
    ) -> np.ndarray:
        return (pseudo_labels == cluster_refined_labels).astype(np.float32)

    def _prototype_similarity(
        self,
        features: np.ndarray,
        pseudo_labels: np.ndarray,
        eps: float = 1e-10,
    ) -> np.ndarray:
        normalized_features = self._l2_normalize(features, eps=eps)
        prototypes = np.zeros(
            (self.num_classes, normalized_features.shape[1]), dtype=np.float32)

        for class_index in range(self.num_classes):
            class_mask = pseudo_labels == class_index
            if class_mask.sum() == 0:
                continue

            class_features = normalized_features[class_mask]
            prototype = class_features.mean(axis=0)
            prototypes[class_index] = prototype / \
                (np.linalg.norm(prototype) + eps)

        similarities = np.zeros(normalized_features.shape[0], dtype=np.float32)

        for index, label in enumerate(pseudo_labels):
            prototype = prototypes[int(label)]
            if np.linalg.norm(prototype) < eps:
                similarities[index] = 0.0
            else:
                similarities[index] = float(
                    np.dot(normalized_features[index], prototype))

        return (similarities + 1.0) / 2.0

    @staticmethod
    def _l2_normalize(features: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        return features / (np.linalg.norm(features, axis=1, keepdims=True) + eps)
