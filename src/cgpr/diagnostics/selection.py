from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from cgpr.diagnostics.scoring import ScoreBank


@dataclass
class SelectionResult:
    selected: dict[int, list[dict]]
    reserve: dict[int, list[dict]]
    rejected: list[dict]
    selected_mask: np.ndarray
    reserve_mask: np.ndarray


class CapacitySelector:
    def __init__(self, capacity_per_class: int, reserve_per_class: int):
        self.capacity_per_class = capacity_per_class
        self.reserve_per_class = reserve_per_class

    def select(self, scores: ScoreBank) -> SelectionResult:
        candidate_class = scores.candidate_class
        candidate_score = scores.candidate_score

        order = np.argsort(-candidate_score)

        selected = defaultdict(list)
        reserve = defaultdict(list)
        rejected = []

        selected_mask = np.zeros(len(candidate_class), dtype=bool)
        reserve_mask = np.zeros(len(candidate_class), dtype=bool)

        for sample_idx in order:
            class_idx = int(candidate_class[sample_idx])

            record = {
                "index": int(sample_idx),
                "class": class_idx,
                "score": float(candidate_score[sample_idx]),
            }

            if len(selected[class_idx]) < self.capacity_per_class:
                selected[class_idx].append(record)
                selected_mask[sample_idx] = True

            elif len(reserve[class_idx]) < self.reserve_per_class:
                reserve[class_idx].append(record)
                reserve_mask[sample_idx] = True

            else:
                rejected.append(record)

        return SelectionResult(
            selected=dict(selected),
            reserve=dict(reserve),
            rejected=rejected,
            selected_mask=selected_mask,
            reserve_mask=reserve_mask,
        )
