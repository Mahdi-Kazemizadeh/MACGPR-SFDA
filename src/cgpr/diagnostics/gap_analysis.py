from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cgpr.diagnostics.audit import AuditResult
from cgpr.diagnostics.clustering import ClusteringResult
from cgpr.diagnostics.config import CapacityAuditConfig, CLASS_NAMES
from cgpr.diagnostics.feature_bank import FeatureBank
from cgpr.diagnostics.prototype import PrototypeBank
from cgpr.diagnostics.reaudit import ReAuditResult
from cgpr.diagnostics.scoring import ScoreBank
from cgpr.diagnostics.selection import SelectionResult


@dataclass
class GapAnalysisResult:
    summary: dict
    raw_paths: dict[str, str]


class GapAnalyzer:
    def __init__(self, config: CapacityAuditConfig):
        self.config = config

    def export(
        self,
        feature_bank: FeatureBank,
        prototype_bank: PrototypeBank,
        clustering: ClusteringResult,
        scores: ScoreBank,
        selection: SelectionResult,
        audit: AuditResult,
        reaudit: ReAuditResult,
    ) -> GapAnalysisResult:
        output_dir = self.config.output_dir / "gap_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        gap_classes = self._find_gap_classes(reaudit.clean)
        membership = self._build_membership(selection, audit, reaudit)

        features_norm = self._normalize(feature_bank.features)
        prototypes_norm = self._normalize(prototype_bank.prototypes)

        summary = {
            "capacity_per_class": self.config.capacity_per_class,
            "gap_classes": {},
        }
        raw_paths = {}

        for class_idx in gap_classes:
            class_name = CLASS_NAMES[class_idx]
            path = output_dir / f"gap_raw_{class_name}.jsonl"

            rows = self._build_rows_for_class(
                class_idx=class_idx,
                feature_bank=feature_bank,
                clustering=clustering,
                scores=scores,
                membership=membership,
                features_norm=features_norm,
                prototypes_norm=prototypes_norm,
            )

            rows = sorted(
                rows,
                key=lambda r: (
                    r["analysis_score"],
                    r["prob_for_gap_class"],
                    r["proto_sim_for_gap_class"],
                    -r["entropy_norm"],
                ),
                reverse=True,
            )

            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            raw_paths[class_name] = str(path)

            summary["gap_classes"][class_name] = self._summarize_class(
                class_idx=class_idx,
                rows=rows,
                reaudit=reaudit,
            )

        summary_path = output_dir / "gap_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Saved gap analysis summary: {summary_path}")
        for class_name, path in raw_paths.items():
            print(f"Saved gap raw data for {class_name}: {path}")

        return GapAnalysisResult(
            summary=summary,
            raw_paths=raw_paths,
        )

    def _find_gap_classes(self, clean: dict[int, list[dict]]) -> list[int]:
        gap_classes = []

        for class_idx in range(self.config.num_classes):
            count = len(clean.get(class_idx, []))
            gap = self.config.capacity_per_class - count

            if gap > 0:
                gap_classes.append(class_idx)

        return gap_classes

    def _build_membership(
        self,
        selection: SelectionResult,
        audit: AuditResult,
        reaudit: ReAuditResult,
    ) -> dict[int, dict]:
        membership = {}

        def add(items: list[dict], key: str, class_idx: int) -> None:
            for item in items:
                sample_idx = int(item["index"])
                if sample_idx not in membership:
                    membership[sample_idx] = {}
                membership[sample_idx][key] = class_idx

        for class_idx in range(self.config.num_classes):
            add(selection.selected.get(class_idx, []), "selected_class", class_idx)
            add(selection.reserve.get(class_idx, []), "reserve_class", class_idx)
            add(audit.clean.get(class_idx, []), "clean_class", class_idx)
            add(audit.suspicious.get(class_idx, []), "suspicious_class", class_idx)
            add(audit.reaudit_pool.get(class_idx, []), "reaudit_pool_class", class_idx)
            add(reaudit.accepted.get(class_idx, []), "reaudit_accepted_class", class_idx)
            add(reaudit.rejected.get(class_idx, []), "reaudit_rejected_class", class_idx)

        return membership

    def _build_rows_for_class(
        self,
        class_idx: int,
        feature_bank: FeatureBank,
        clustering: ClusteringResult,
        scores: ScoreBank,
        membership: dict[int, dict],
        features_norm: np.ndarray,
        prototypes_norm: np.ndarray,
    ) -> list[dict]:
        rows = []
        num_samples = feature_bank.features.shape[0]
        entropy_norm = feature_bank.entropy / np.log(self.config.num_classes)

        class_probs = feature_bank.probs[:, class_idx]
        proto_sims = features_norm @ prototypes_norm[class_idx]
        proto_dists = 1.0 - proto_sims

        rival_probs = feature_bank.probs.copy()
        rival_probs[:, class_idx] = -1.0
        max_rival_probs = rival_probs.max(axis=1)
        class_margin = class_probs - max_rival_probs

        prob_ranks = self._prob_ranks(feature_bank.probs, class_idx)

        analysis_score = (
            0.40 * class_probs
            + 0.35 * proto_sims
            + 0.15 * np.maximum(class_margin, 0.0)
            + 0.10 * (1.0 - entropy_norm)
            - 0.15 * max_rival_probs
        )

        for sample_idx in range(num_samples):
            cluster_idx = int(clustering.cluster_labels[sample_idx])
            mapped_class = int(clustering.cluster_to_class[cluster_idx]) if cluster_idx >= 0 else -1

            member = membership.get(sample_idx, {})

            row = {
                "index": int(sample_idx),

                "gap_class_index": int(class_idx),
                "gap_class_name": CLASS_NAMES[class_idx],

                "true_label_index_debug": int(feature_bank.true_labels[sample_idx]),
                "true_label_name_debug": CLASS_NAMES[int(feature_bank.true_labels[sample_idx])],

                "source_pseudo_index": int(feature_bank.pseudo[sample_idx]),
                "source_pseudo_name": CLASS_NAMES[int(feature_bank.pseudo[sample_idx])],
                "source_confidence": float(feature_bank.confidence[sample_idx]),

                "candidate_class_index": int(scores.candidate_class[sample_idx]),
                "candidate_class_name": CLASS_NAMES[int(scores.candidate_class[sample_idx])],
                "candidate_score": float(scores.candidate_score[sample_idx]),

                "prob_for_gap_class": float(class_probs[sample_idx]),
                "prob_rank_for_gap_class": int(prob_ranks[sample_idx]),
                "max_rival_prob": float(max_rival_probs[sample_idx]),
                "class_margin_vs_max_rival": float(class_margin[sample_idx]),

                "proto_sim_for_gap_class": float(proto_sims[sample_idx]),
                "proto_dist_for_gap_class": float(proto_dists[sample_idx]),

                "entropy": float(feature_bank.entropy[sample_idx]),
                "entropy_norm": float(entropy_norm[sample_idx]),

                "score_for_gap_class": float(scores.score[sample_idx, class_idx]),
                "analysis_score": float(analysis_score[sample_idx]),

                "cluster_id": int(cluster_idx),
                "cluster_mapped_class_index": int(mapped_class),
                "cluster_mapped_class_name": CLASS_NAMES[mapped_class] if mapped_class >= 0 else "unknown",
                "cluster_agrees_with_gap_class": bool(mapped_class == class_idx),

                "selected_class": self._class_name_or_none(member.get("selected_class")),
                "reserve_class": self._class_name_or_none(member.get("reserve_class")),
                "clean_class": self._class_name_or_none(member.get("clean_class")),
                "suspicious_class": self._class_name_or_none(member.get("suspicious_class")),
                "reaudit_pool_class": self._class_name_or_none(member.get("reaudit_pool_class")),
                "reaudit_accepted_class": self._class_name_or_none(member.get("reaudit_accepted_class")),
                "reaudit_rejected_class": self._class_name_or_none(member.get("reaudit_rejected_class")),
            }

            rows.append(row)

        return rows

    def _summarize_class(
        self,
        class_idx: int,
        rows: list[dict],
        reaudit: ReAuditResult,
    ) -> dict:
        current_count = len(reaudit.clean.get(class_idx, []))
        gap = max(0, self.config.capacity_per_class - current_count)

        thresholds = {
            "prob_ge_0_05": 0.05,
            "prob_ge_0_10": 0.10,
            "prob_ge_0_20": 0.20,
            "prob_ge_0_30": 0.30,
            "prob_ge_0_40": 0.40,
        }

        out = {
            "current_count": int(current_count),
            "gap": int(gap),
            "total_target_samples": int(len(rows)),
            "top20": rows[:20],
            "threshold_counts": {},
            "rank_counts": {},
            "cluster_agreement_count": int(sum(r["cluster_agrees_with_gap_class"] for r in rows)),
        }

        for name, thr in thresholds.items():
            out["threshold_counts"][name] = int(
                sum(r["prob_for_gap_class"] >= thr for r in rows)
            )

        for rank in [1, 2, 3, 5, 10, 12]:
            out["rank_counts"][f"rank_le_{rank}"] = int(
                sum(r["prob_rank_for_gap_class"] <= rank for r in rows)
            )

        return out

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    @staticmethod
    def _prob_ranks(probs: np.ndarray, class_idx: int) -> np.ndarray:
        order = np.argsort(-probs, axis=1)
        ranks = np.zeros(probs.shape[0], dtype=np.int64)

        for i in range(probs.shape[0]):
            ranks[i] = int(np.where(order[i] == class_idx)[0][0]) + 1

        return ranks

    @staticmethod
    def _class_name_or_none(class_idx: int | None) -> str | None:
        if class_idx is None:
            return None
        return CLASS_NAMES[int(class_idx)]
