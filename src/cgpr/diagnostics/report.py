from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cgpr.diagnostics.audit import AuditResult
from cgpr.diagnostics.clustering import ClusteringResult
from cgpr.diagnostics.config import CapacityAuditConfig, CLASS_NAMES
from cgpr.diagnostics.feature_bank import FeatureBank
from cgpr.diagnostics.prototype import PrototypeBank
from cgpr.diagnostics.reaudit import ReAuditResult
from cgpr.diagnostics.replacement import ReplacementResult
from cgpr.diagnostics.scoring import ScoreBank
from cgpr.diagnostics.selection import SelectionResult


class ReportBuilder:
    def __init__(self, config: CapacityAuditConfig):
        self.config = config

    def build(
        self,
        feature_bank: FeatureBank,
        prototype_bank: PrototypeBank,
        clustering: ClusteringResult,
        scores: ScoreBank,
        selection: SelectionResult,
        audit: AuditResult,
        reaudit: ReAuditResult,
        replacement: ReplacementResult,
    ) -> dict[str, Any]:
        rows = []

        for class_idx in range(self.config.num_classes):
            initial_idx = self._indices(selection.selected.get(class_idx, []))
            reserve_idx = self._indices(selection.reserve.get(class_idx, []))
            clean_idx = self._indices(audit.clean.get(class_idx, []))
            suspicious_idx = self._indices(audit.suspicious.get(class_idx, []))
            reaudit_pool_idx = self._indices(audit.reaudit_pool.get(class_idx, []))
            reaudit_accepted_idx = self._indices(reaudit.accepted.get(class_idx, []))
            final_idx = self._indices(replacement.final_selected.get(class_idx, []))
            repl_idx = self._indices(replacement.replacements.get(class_idx, []))

            row = {
                "class_index": class_idx,
                "class_name": CLASS_NAMES[class_idx],
                "selected_initial": len(initial_idx),
                "reserve": len(reserve_idx),
                "clean_after_audit": len(clean_idx),
                "suspicious_removed": len(suspicious_idx),
                "reaudit_pool": len(reaudit_pool_idx),
                "reaudit_accepted": len(reaudit_accepted_idx),
                "replacements": len(repl_idx),
                "final_selected": len(final_idx),
                "gap": max(0, self.config.capacity_per_class - len(final_idx)),
                "mean_conf_initial": self._mean_or_none(feature_bank.confidence, initial_idx),
                "mean_conf_final": self._mean_or_none(feature_bank.confidence, final_idx),
                "mean_entropy_final": self._mean_or_none(feature_bank.entropy, final_idx),
                "mean_margin_final": self._mean_or_none(scores.margin, final_idx),
                "mean_proto_dist_final": self._mean_or_none(scores.own_dist, final_idx),
            }

            if self.config.debug_labels:
                row.update(self._debug_class_metrics(feature_bank, final_idx, class_idx))

            rows.append(row)

        global_info = {
            "capacity_per_class": self.config.capacity_per_class,
            "reserve_per_class": self.config.reserve_per_class,
            "total_initial_selected": int(sum(len(v) for v in selection.selected.values())),
            "total_reserve": int(sum(len(v) for v in selection.reserve.values())),
            "total_rejected": int(len(selection.rejected)),
            "total_clean_after_audit": int(sum(len(v) for v in audit.clean.values())),
            "total_suspicious_removed": int(sum(len(v) for v in audit.suspicious.values())),
            "total_reaudit_pool": int(sum(len(v) for v in audit.reaudit_pool.values())),
            "total_reaudit_accepted": int(sum(len(v) for v in reaudit.accepted.values())),
            "total_replacements": int(sum(len(v) for v in replacement.replacements.values())),
            "total_final_selected": int(sum(len(v) for v in replacement.final_selected.values())),
            "total_gap": int(
                sum(
                    max(
                        0,
                        self.config.capacity_per_class
                        - len(replacement.final_selected.get(class_idx, [])),
                    )
                    for class_idx in range(self.config.num_classes)
                )
            ),
        }

        if self.config.debug_labels:
            global_info.update(self._debug_global_metrics(feature_bank, replacement))

        report = {
            "global": global_info,
            "per_class": rows,
            "prototype_source_counts": {
                CLASS_NAMES[i]: int(prototype_bank.source_counts[i])
                for i in range(self.config.num_classes)
            },
            "cluster_to_class": {
                str(i): CLASS_NAMES[int(c)] if int(c) >= 0 else "unknown"
                for i, c in enumerate(clustering.cluster_to_class)
            },
        }

        return report

    @staticmethod
    def _indices(items: list[dict]) -> list[int]:
        return [int(item["index"]) for item in items]

    @staticmethod
    def _mean_or_none(values: np.ndarray, indices: list[int]) -> float | None:
        if len(indices) == 0:
            return None
        return float(np.mean(values[indices]))

    def _debug_class_metrics(
        self,
        feature_bank: FeatureBank,
        final_indices: list[int],
        class_idx: int,
    ) -> dict[str, Any]:
        if len(final_indices) == 0:
            return {
                "debug_final_true_majority": None,
                "debug_final_true_majority_name": None,
                "debug_final_purity": None,
            }

        final_true = feature_bank.true_labels[final_indices]
        majority = int(np.bincount(final_true, minlength=self.config.num_classes).argmax())

        return {
            "debug_final_true_majority": majority,
            "debug_final_true_majority_name": CLASS_NAMES[majority],
            "debug_final_purity": float((final_true == class_idx).mean()),
        }

    def _debug_global_metrics(
        self,
        feature_bank: FeatureBank,
        replacement: ReplacementResult,
    ) -> dict[str, Any]:
        final_indices = []
        assigned = []

        for class_idx in range(self.config.num_classes):
            items = replacement.final_selected.get(class_idx, [])
            indices = self._indices(items)
            final_indices.extend(indices)
            assigned.extend([class_idx] * len(indices))

        if len(final_indices) == 0:
            return {"debug_final_accuracy_vs_folder_labels": None}

        assigned_np = np.asarray(assigned, dtype=np.int64)
        true_np = feature_bank.true_labels[final_indices]

        return {
            "debug_final_accuracy_vs_folder_labels": float((assigned_np == true_np).mean())
        }

    def print_report(self, report: dict[str, Any]) -> None:
        print("\n" + "=" * 110)
        print("Capacity-Aware Audit Diagnostic")
        print("=" * 110)
        print(json.dumps(report["global"], indent=2))

        print("\nPer-class summary:")
        header = (
            f"{'class':<12} {'init':>6} {'res':>6} {'clean':>6} "
            f"{'susp':>6} {'reaud':>6} {'racc':>6} {'repl':>6} "
            f"{'final':>6} {'gap':>6} {'conf':>8} {'margin':>8} {'dist':>8}"
        )
        print(header)
        print("-" * len(header))

        for row in report["per_class"]:
            print(
                f"{row['class_name']:<12} "
                f"{row['selected_initial']:>6} "
                f"{row['reserve']:>6} "
                f"{row['clean_after_audit']:>6} "
                f"{row['suspicious_removed']:>6} "
                f"{row['reaudit_pool']:>6} "
                f"{row['reaudit_accepted']:>6} "
                f"{row['replacements']:>6} "
                f"{row['final_selected']:>6} "
                f"{row['gap']:>6} "
                f"{self._fmt(row['mean_conf_final']):>8} "
                f"{self._fmt(row['mean_margin_final']):>8} "
                f"{self._fmt(row['mean_proto_dist_final']):>8}"
            )

        acc = report["global"].get("debug_final_accuracy_vs_folder_labels")
        if acc is not None:
            print("\nDebug label-aware final accuracy:")
            print(f"{acc:.4f}")

    @staticmethod
    def _fmt(value: float | None) -> str:
        if value is None:
            return "0.000"
        return f"{value:.3f}"

    def save(
        self,
        report: dict[str, Any],
        audit: AuditResult,
        reaudit: ReAuditResult,
        replacement: ReplacementResult,
    ) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self._save_json("capacity_audit_report.json", report)

        self._save_json(
            "capacity_audit_final_selected.json",
            self._by_class_name(replacement.final_selected),
        )

        self._save_json(
            "capacity_audit_suspicious.json",
            self._by_class_name(audit.suspicious),
        )

        self._save_json(
            "capacity_audit_reaudit_pool.json",
            self._by_class_name(audit.reaudit_pool),
        )

        self._save_json(
            "capacity_audit_reaudit_accepted.json",
            self._by_class_name(reaudit.accepted),
        )

        self._save_json(
            "capacity_audit_reaudit_rejected.json",
            self._by_class_name(reaudit.rejected),
        )

        self._save_json(
            "capacity_audit_reaudit_clusters.json",
            self._by_class_name(reaudit.cluster_reports),
        )

        self._save_json(
            "capacity_audit_replacements.json",
            self._by_class_name(replacement.replacements),
        )

        print(f"Saved outputs to: {self.config.output_dir}")

    def _save_json(self, filename: str, payload: Any) -> None:
        path = self.config.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _by_class_name(self, data: dict[int, list[dict]]) -> dict[str, list[dict]]:
        return {
            CLASS_NAMES[class_idx]: data.get(class_idx, [])
            for class_idx in range(self.config.num_classes)
        }
