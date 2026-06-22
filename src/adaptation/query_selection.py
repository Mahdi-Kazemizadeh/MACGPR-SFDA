from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class QuerySelectionResult:
    """Stores masks for selective pseudo-label usage and MLLM querying."""

    easy_mask: np.ndarray
    hard_mask: np.ndarray
    unsafe_mask: np.ndarray
    reliability_scores: np.ndarray

    @property
    def easy_count(self) -> int:
        return int(self.easy_mask.sum())

    @property
    def hard_count(self) -> int:
        return int(self.hard_mask.sum())

    @property
    def unsafe_count(self) -> int:
        return int(self.unsafe_mask.sum())


class QuerySelector:
    """Selects easy, hard, and unsafe samples without using target labels."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        query_config = config["adaptation"].get("query_selection", {})

        self.easy_threshold = float(query_config.get("easy_threshold", 0.75))
        self.hard_lower_threshold = float(
            query_config.get("hard_lower_threshold", 0.45))
        self.hard_upper_threshold = float(
            query_config.get("hard_upper_threshold", 0.75))
        self.max_hard_ratio = float(query_config.get("max_hard_ratio", 0.20))

    def select(self, reliability_scores: np.ndarray) -> QuerySelectionResult:
        """Split target samples into easy, hard, and unsafe groups.

        Easy samples are used directly for pseudo-label training.
        Hard samples are candidates for MLLM verification.
        Unsafe samples are ignored.
        """
        easy_mask = reliability_scores >= self.easy_threshold

        hard_mask = (
            (reliability_scores >= self.hard_lower_threshold)
            & (reliability_scores < self.hard_upper_threshold)
        )

        hard_mask = self._limit_hard_samples(
            reliability_scores=reliability_scores,
            hard_mask=hard_mask,
        )

        unsafe_mask = ~(easy_mask | hard_mask)

        return QuerySelectionResult(
            easy_mask=easy_mask,
            hard_mask=hard_mask,
            unsafe_mask=unsafe_mask,
            reliability_scores=reliability_scores,
        )

    def _limit_hard_samples(
        self,
        reliability_scores: np.ndarray,
        hard_mask: np.ndarray,
    ) -> np.ndarray:
        max_hard_samples = int(len(reliability_scores) * self.max_hard_ratio)
        hard_indices = np.where(hard_mask)[0]

        if len(hard_indices) <= max_hard_samples:
            return hard_mask

        hard_scores = reliability_scores[hard_indices]

        order = np.argsort(hard_scores)
        selected_hard_indices = hard_indices[order[:max_hard_samples]]

        limited_hard_mask = np.zeros_like(hard_mask, dtype=bool)
        limited_hard_mask[selected_hard_indices] = True

        return limited_hard_mask
