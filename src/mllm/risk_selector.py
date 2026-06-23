from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class RiskSelectionResult:
    accept_mask: np.ndarray
    query_mask: np.ndarray
    hold_mask: np.ndarray
    reject_mask: np.ndarray
    decision: List[str]
    reason: List[str]
    prototype_labels: np.ndarray
    cluster_majority_labels: np.ndarray
    dead_classes: List[int]
    dominant_classes: List[int]
    stats: Dict[str, int]


class RiskAwareSelector:
    """
    Risk-aware selector for conditional MLLM invocation.

    This module does NOT assume that low reliability means wrong.
    It only decides whether the internal SFDA model can safely use a sample,
    whether the sample should be sent to an MLLM verifier, or whether it should
    be held/ignored.

    MLLM is invoked only for structurally risky samples.
    """

    def __init__(
        self,
        num_classes: int,
        high_confidence: float = 0.95,
        low_confidence: float = 0.45,
        low_entropy: float = 0.25,
        high_entropy: float = 0.75,
        safe_reliability: float = 0.95,
        query_reliability: float = 0.80,
        min_class_ratio: float = 0.005,
        dominant_class_ratio: float = 0.18,
        prototype_min_count: int = 20,
        prototype_reliability_threshold: float = 0.90,
    ) -> None:
        self.num_classes = num_classes
        self.high_confidence = high_confidence
        self.low_confidence = low_confidence
        self.low_entropy = low_entropy
        self.high_entropy = high_entropy
        self.safe_reliability = safe_reliability
        self.query_reliability = query_reliability
        self.min_class_ratio = min_class_ratio
        self.dominant_class_ratio = dominant_class_ratio
        self.prototype_min_count = prototype_min_count
        self.prototype_reliability_threshold = prototype_reliability_threshold

    def select(
        self,
        normalized_features: np.ndarray,
        probabilities: np.ndarray,
        pseudo_labels: np.ndarray,
        reliability_scores: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> RiskSelectionResult:
        num_samples = len(pseudo_labels)

        confidences = probabilities.max(axis=1)
        entropies = self._normalized_entropy(probabilities)

        class_counts = np.bincount(pseudo_labels, minlength=self.num_classes)
        class_ratios = class_counts / max(num_samples, 1)

        dead_classes = np.where(class_ratios < self.min_class_ratio)[
            0].tolist()
        dominant_classes = np.where(
            class_ratios > self.dominant_class_ratio)[0].tolist()

        prototype_labels = self._nearest_prototype_labels(
            normalized_features=normalized_features,
            pseudo_labels=pseudo_labels,
            reliability_scores=reliability_scores,
        )

        cluster_majority_labels = self._cluster_majority_labels(
            cluster_labels=cluster_labels,
            pseudo_labels=pseudo_labels,
        )

        accept_mask = np.zeros(num_samples, dtype=bool)
        query_mask = np.zeros(num_samples, dtype=bool)
        hold_mask = np.zeros(num_samples, dtype=bool)
        reject_mask = np.zeros(num_samples, dtype=bool)

        decisions: List[str] = []
        reasons: List[str] = []

        collapse_alarm = len(dead_classes) > 0

        for i in range(num_samples):
            pseudo = int(pseudo_labels[i])
            conf = float(confidences[i])
            ent = float(entropies[i])
            rel = float(reliability_scores[i])

            proto_label = int(prototype_labels[i])
            cluster_label = int(cluster_majority_labels[i])

            prototype_agree = proto_label == pseudo
            cluster_agree = cluster_label == pseudo

            structural_agreement = prototype_agree and cluster_agree
            structural_conflict = (not prototype_agree) or (not cluster_agree)

            high_conf_suspicious = (
                conf >= self.high_confidence
                and structural_conflict
            )

            low_reliability_but_informative = (
                rel < self.query_reliability
                and structural_conflict
                and conf >= self.low_confidence
                and conf < self.high_confidence
                and ent < self.high_entropy
            )

            collapse_sensitive = (
                collapse_alarm
                and pseudo in dominant_classes
                and (
                    structural_conflict
                    or rel < self.query_reliability
                    or ent >= self.high_entropy
                )
            )

            safe_without_mllm = (
                conf >= self.high_confidence
                and ent <= self.low_entropy
                and rel >= self.safe_reliability
                and structural_agreement
                and not collapse_sensitive
            )

            very_noisy = (
                conf <= self.low_confidence
                and ent >= self.high_entropy
                and rel < self.query_reliability
                and not structural_conflict
            )

            if safe_without_mllm:
                accept_mask[i] = True
                decisions.append("ACCEPT_WITHOUT_MLLM")
                reasons.append("safe_structural_agreement")

            elif high_conf_suspicious:
                query_mask[i] = True
                decisions.append("QUERY_MLLM")
                reasons.append("high_confidence_structural_conflict")

            elif collapse_sensitive:
                query_mask[i] = True
                decisions.append("QUERY_MLLM")
                reasons.append("collapse_sensitive_dominant_class")

            elif low_reliability_but_informative:
                query_mask[i] = True
                decisions.append("QUERY_MLLM")
                reasons.append("low_reliability_structural_conflict")

            elif very_noisy:
                hold_mask[i] = True
                decisions.append("HOLD")
                reasons.append("very_noisy_not_safe_for_training")

            else:
                hold_mask[i] = True
                decisions.append("HOLD")
                reasons.append("uncertain_hold_for_later")

        stats = {
            "num_accept_without_mllm": int(accept_mask.sum()),
            "num_query_mllm": int(query_mask.sum()),
            "num_hold": int(hold_mask.sum()),
            "num_reject": int(reject_mask.sum()),
            "num_dead_classes": len(dead_classes),
            "num_dominant_classes": len(dominant_classes),
        }

        return RiskSelectionResult(
            accept_mask=accept_mask,
            query_mask=query_mask,
            hold_mask=hold_mask,
            reject_mask=reject_mask,
            decision=decisions,
            reason=reasons,
            prototype_labels=prototype_labels,
            cluster_majority_labels=cluster_majority_labels,
            dead_classes=dead_classes,
            dominant_classes=dominant_classes,
            stats=stats,
        )

    def _normalized_entropy(self, probabilities: np.ndarray) -> np.ndarray:
        eps = 1e-12
        entropy = -(probabilities * np.log(probabilities + eps)).sum(axis=1)
        max_entropy = np.log(self.num_classes)
        return entropy / max(max_entropy, eps)

    def _nearest_prototype_labels(
        self,
        normalized_features: np.ndarray,
        pseudo_labels: np.ndarray,
        reliability_scores: np.ndarray,
    ) -> np.ndarray:
        feature_dim = normalized_features.shape[1]
        prototypes = np.zeros(
            (self.num_classes, feature_dim), dtype=np.float32)
        valid_prototypes = np.zeros(self.num_classes, dtype=bool)

        for class_id in range(self.num_classes):
            reliable_mask = (
                (pseudo_labels == class_id)
                & (reliability_scores >= self.prototype_reliability_threshold)
            )

            if reliable_mask.sum() < self.prototype_min_count:
                reliable_mask = pseudo_labels == class_id

            if reliable_mask.sum() == 0:
                continue

            prototype = normalized_features[reliable_mask].mean(axis=0)
            norm = np.linalg.norm(prototype) + 1e-12
            prototypes[class_id] = prototype / norm
            valid_prototypes[class_id] = True

        similarities = normalized_features @ prototypes.T

        invalid_classes = np.where(~valid_prototypes)[0]
        if len(invalid_classes) > 0:
            similarities[:, invalid_classes] = -1e9

        nearest_labels = similarities.argmax(axis=1)

        return nearest_labels.astype(np.int64)

    def _cluster_majority_labels(
        self,
        cluster_labels: np.ndarray,
        pseudo_labels: np.ndarray,
    ) -> np.ndarray:
        cluster_majority: Dict[int, int] = {}

        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            labels = pseudo_labels[mask]

            if len(labels) == 0:
                cluster_majority[int(cluster_id)] = -1
                continue

            counts = np.bincount(labels, minlength=self.num_classes)
            cluster_majority[int(cluster_id)] = int(counts.argmax())

        majority_labels = np.array(
            [cluster_majority[int(c)] for c in cluster_labels],
            dtype=np.int64,
        )

        return majority_labels
