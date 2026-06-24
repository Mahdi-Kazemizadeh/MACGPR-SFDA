from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from cgpr.diagnostics.config import CapacityAuditConfig
from cgpr.diagnostics.feature_bank import FeatureBank
from cgpr.diagnostics.clustering import ClusteringResult
from cgpr.diagnostics.scoring import ScoreBank
from cgpr.diagnostics.selection import SelectionResult


@dataclass
class AuditResult:
    clean: dict[int, list[dict]]
    suspicious: dict[int, list[dict]]
    reaudit_pool: dict[int, list[dict]]


class AuditEngine:
    def __init__(self, config: CapacityAuditConfig):
        self.config = config

    def audit(
        self,
        selection: SelectionResult,
        feature_bank: FeatureBank,
        clustering: ClusteringResult,
        scores: ScoreBank,
    ) -> AuditResult:
        clean = defaultdict(list)
        suspicious = defaultdict(list)
        reaudit_pool = defaultdict(list)

        confidence = feature_bank.confidence
        pseudo = feature_bank.pseudo

        cluster_labels = clustering.cluster_labels
        cluster_to_class = clustering.cluster_to_class

        candidate_class = scores.candidate_class
        margin = scores.margin
        own_dist = scores.own_dist

        for class_idx, items in selection.selected.items():
            for item in items:
                sample_idx = int(item["index"])
                reasons = []

                if confidence[sample_idx] < self.config.audit_min_confidence:
                    reasons.append("low_confidence")

                if margin[sample_idx] < self.config.audit_min_margin:
                    reasons.append("low_margin")

                if own_dist[sample_idx] > self.config.audit_max_proto_dist:
                    reasons.append("far_from_candidate_prototype")

                cluster_idx = int(cluster_labels[sample_idx])
                mapped_class = int(cluster_to_class[cluster_idx]) if cluster_idx >= 0 else -1

                if mapped_class >= 0 and mapped_class != class_idx:
                    reasons.append("cluster_disagreement")

                if (
                    int(pseudo[sample_idx]) != class_idx
                    and confidence[sample_idx] >= self.config.audit_pseudo_conflict_confidence
                ):
                    reasons.append("confident_pseudo_conflict")

                if int(candidate_class[sample_idx]) != class_idx:
                    reasons.append("candidate_class_changed")

                cluster_only = reasons == ["cluster_disagreement"]
                hard_reasons = [reason for reason in reasons if reason != "cluster_disagreement"]

                item_out = dict(item)

                if cluster_only:
                    item_out["reasons"] = reasons
                    item_out["reaudit_reason"] = "cluster_disagreement_only"
                    reaudit_pool[class_idx].append(item_out)

                elif hard_reasons:
                    item_out["reasons"] = reasons
                    suspicious[class_idx].append(item_out)

                else:
                    clean[class_idx].append(item_out)

        return AuditResult(
            clean=dict(clean),
            suspicious=dict(suspicious),
            reaudit_pool=dict(reaudit_pool),
        )
