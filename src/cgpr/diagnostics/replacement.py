from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from cgpr.diagnostics.config import CapacityAuditConfig
from cgpr.diagnostics.feature_bank import FeatureBank
from cgpr.diagnostics.prototype import PrototypeBank
from cgpr.diagnostics.clustering import ClusteringResult
from cgpr.diagnostics.audit import AuditResult
from cgpr.diagnostics.reaudit import ReAuditResult
from cgpr.diagnostics.selection import SelectionResult


@dataclass
class ReplacementResult:
    final_selected: dict[int, list[dict]]
    replacements: dict[int, list[dict]]
    failed_replacements: dict[int, list[dict]]


class ReplacementEngine:
    def __init__(self, config: CapacityAuditConfig):
        self.config = config

    def run(
        self,
        selection: SelectionResult,
        audit: AuditResult,
        reaudit: ReAuditResult,
        feature_bank: FeatureBank,
        prototype_bank: PrototypeBank,
        clustering: ClusteringResult,
    ) -> ReplacementResult:
        clean_prototypes = self._recompute_clean_prototypes(
            clean=reaudit.clean,
            feature_bank=feature_bank,
            prototype_bank=prototype_bank,
        )

        final_selected = {
            class_idx: list(reaudit.clean.get(class_idx, []))
            for class_idx in range(self.config.num_classes)
        }

        replacements = defaultdict(list)
        failed_replacements = defaultdict(list)

        used = set()
        for items in final_selected.values():
            for item in items:
                used.add(int(item["index"]))

        for class_idx in range(self.config.num_classes):
            gap = self.config.capacity_per_class - len(final_selected[class_idx])

            if gap <= 0:
                continue

            candidates = sorted(
                selection.reserve.get(class_idx, []),
                key=lambda item: float(item["score"]),
                reverse=True,
            )

            for tier in range(self.config.replacement_tiers):
                if gap <= 0:
                    break

                for item in candidates:
                    if gap <= 0:
                        break

                    sample_idx = int(item["index"])

                    if sample_idx in used:
                        continue

                    ok, info = self._validate_replacement(
                        sample_idx=sample_idx,
                        class_idx=class_idx,
                        tier=tier,
                        feature_bank=feature_bank,
                        clean_prototypes=clean_prototypes,
                        clustering=clustering,
                    )

                    if ok:
                        record = dict(item)
                        record.update(info)
                        record["source"] = "reserve_replacement"
                        final_selected[class_idx].append(record)
                        replacements[class_idx].append(record)
                        used.add(sample_idx)
                        gap -= 1
                    else:
                        if tier == self.config.replacement_tiers - 1:
                            record = dict(item)
                            record.update(info)
                            failed_replacements[class_idx].append(record)

        return ReplacementResult(
            final_selected=final_selected,
            replacements=dict(replacements),
            failed_replacements=dict(failed_replacements),
        )

    def _recompute_clean_prototypes(
        self,
        clean: dict[int, list[dict]],
        feature_bank: FeatureBank,
        prototype_bank: PrototypeBank,
    ) -> np.ndarray:
        output = prototype_bank.prototypes.copy()

        for class_idx in range(self.config.num_classes):
            indices = [int(item["index"]) for item in clean.get(class_idx, [])]

            if len(indices) >= 3:
                prototype = feature_bank.features[indices].mean(axis=0)
                prototype = prototype / (np.linalg.norm(prototype) + 1e-8)
                output[class_idx] = prototype

        return output

    def _validate_replacement(
        self,
        sample_idx: int,
        class_idx: int,
        tier: int,
        feature_bank: FeatureBank,
        clean_prototypes: np.ndarray,
        clustering: ClusteringResult,
    ) -> tuple[bool, dict]:
        cosine = feature_bank.features[sample_idx] @ clean_prototypes.T
        dist = 1.0 - cosine

        order = np.argsort(dist)
        nearest_class = int(order[0])
        nearest_dist = float(dist[nearest_class])
        second_dist = float(dist[order[1]])
        margin = second_dist - nearest_dist
        own_dist = float(dist[class_idx])

        if tier == 0:
            min_conf = self.config.replace_min_confidence
            min_margin = self.config.replace_min_margin
            max_dist = self.config.replace_max_proto_dist
        elif tier == 1:
            min_conf = max(0.0, self.config.replace_min_confidence - 0.10)
            min_margin = max(0.0, self.config.replace_min_margin * 0.50)
            max_dist = self.config.replace_max_proto_dist + 0.05
        else:
            min_conf = max(0.0, self.config.replace_min_confidence - 0.20)
            min_margin = 0.0
            max_dist = self.config.replace_max_proto_dist + 0.10

        reasons = []

        if feature_bank.confidence[sample_idx] < min_conf:
            reasons.append("low_confidence")

        if own_dist > max_dist:
            reasons.append("far_from_clean_prototype")

        if margin < min_margin:
            reasons.append("low_margin")

        cluster_idx = int(clustering.cluster_labels[sample_idx])
        mapped_class = int(clustering.cluster_to_class[cluster_idx]) if cluster_idx >= 0 else -1

        if mapped_class >= 0 and mapped_class != class_idx and tier < 2:
            reasons.append("cluster_disagreement")

        if nearest_class != class_idx and tier < 2:
            reasons.append("nearest_class_not_candidate")

        if (
            int(feature_bank.pseudo[sample_idx]) != class_idx
            and feature_bank.confidence[sample_idx] >= self.config.replace_pseudo_conflict_confidence
            and tier < 2
        ):
            reasons.append("confident_pseudo_conflict")

        ok = len(reasons) == 0

        return ok, {
            "own_dist": float(own_dist),
            "nearest_class": int(nearest_class),
            "nearest_dist": float(nearest_dist),
            "margin": float(margin),
            "tier": int(tier),
            "reasons": reasons,
        }
