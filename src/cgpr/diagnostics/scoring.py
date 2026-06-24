from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from cgpr.diagnostics.clustering import ClusteringResult
from cgpr.diagnostics.feature_bank import FeatureBank
from cgpr.diagnostics.prototype import PrototypeBank


@dataclass
class ScoreBank:
    cosine: np.ndarray
    proto_dist: np.ndarray
    proto_sim: np.ndarray
    cluster_bonus: np.ndarray
    score: np.ndarray
    candidate_class: np.ndarray
    candidate_score: np.ndarray
    nearest_dist: np.ndarray
    second_dist: np.ndarray
    margin: np.ndarray
    own_dist: np.ndarray


class ScoreComputer:
    def compute(
        self,
        feature_bank: FeatureBank,
        prototype_bank: PrototypeBank,
        clustering: ClusteringResult,
    ) -> ScoreBank:
        features = feature_bank.features
        probs = feature_bank.probs
        confidence = feature_bank.confidence
        entropy = feature_bank.entropy
        prototypes = prototype_bank.prototypes

        num_classes = probs.shape[1]

        cosine = features @ prototypes.T
        proto_dist = 1.0 - cosine
        proto_sim = (cosine + 1.0) / 2.0

        entropy_norm = entropy / math.log(num_classes)

        cluster_bonus = np.zeros_like(probs, dtype=np.float32)

        for sample_idx in range(features.shape[0]):
            cluster_idx = int(clustering.cluster_labels[sample_idx])

            if cluster_idx < 0:
                continue

            mapped_class = int(clustering.cluster_to_class[cluster_idx])

            if mapped_class >= 0:
                cluster_bonus[sample_idx, mapped_class] = 1.0

        score = (
            0.45 * probs
            + 0.30 * proto_sim
            + 0.15 * cluster_bonus
            + 0.10 * confidence[:, None]
            - 0.10 * entropy_norm[:, None]
        )

        candidate_class = score.argmax(axis=1)
        candidate_score = score[np.arange(score.shape[0]), candidate_class]

        sorted_dist = np.sort(proto_dist, axis=1)
        nearest_dist = sorted_dist[:, 0]
        second_dist = sorted_dist[:, 1]
        margin = second_dist - nearest_dist

        own_dist = proto_dist[np.arange(proto_dist.shape[0]), candidate_class]

        return ScoreBank(
            cosine=cosine.astype(np.float32),
            proto_dist=proto_dist.astype(np.float32),
            proto_sim=proto_sim.astype(np.float32),
            cluster_bonus=cluster_bonus.astype(np.float32),
            score=score.astype(np.float32),
            candidate_class=candidate_class.astype(np.int64),
            candidate_score=candidate_score.astype(np.float32),
            nearest_dist=nearest_dist.astype(np.float32),
            second_dist=second_dist.astype(np.float32),
            margin=margin.astype(np.float32),
            own_dist=own_dist.astype(np.float32),
        )
