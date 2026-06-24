from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cgpr.diagnostics.feature_bank import FeatureBank


@dataclass
class ClusteringResult:
    cluster_labels: np.ndarray
    cluster_to_class: np.ndarray


class ClusterAnalyzer:
    def __init__(self, num_classes: int, seed: int):
        self.num_classes = num_classes
        self.seed = seed

    def run(self, bank: FeatureBank) -> ClusteringResult:
        try:
            from sklearn.cluster import MiniBatchKMeans
        except Exception as exc:
            print("sklearn is not available. Cluster agreement will be disabled.")
            print(f"Import error: {exc}")

            return ClusteringResult(
                cluster_labels=np.full(bank.features.shape[0], -1, dtype=np.int64),
                cluster_to_class=np.full(self.num_classes, -1, dtype=np.int64),
            )

        print("Running MiniBatchKMeans...")

        kmeans = MiniBatchKMeans(
            n_clusters=self.num_classes,
            random_state=self.seed,
            batch_size=4096,
            n_init="auto",
            max_iter=100,
            reassignment_ratio=0.01,
        )

        cluster_labels = kmeans.fit_predict(bank.features)
        cluster_to_class = np.full(self.num_classes, -1, dtype=np.int64)

        for cluster_idx in range(self.num_classes):
            indices = np.where(cluster_labels == cluster_idx)[0]

            if len(indices) == 0:
                continue

            votes = np.zeros(self.num_classes, dtype=np.float64)

            for sample_idx in indices:
                pseudo_class = int(bank.pseudo[sample_idx])
                votes[pseudo_class] += float(bank.confidence[sample_idx])

            cluster_to_class[cluster_idx] = int(votes.argmax())

        return ClusteringResult(
            cluster_labels=cluster_labels.astype(np.int64),
            cluster_to_class=cluster_to_class.astype(np.int64),
        )
