from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cgpr.diagnostics.config import CapacityAuditConfig, CLASS_NAMES
from cgpr.diagnostics.feature_bank import FeatureBank
from cgpr.diagnostics.prototype import PrototypeBank
from cgpr.diagnostics.audit import AuditResult
from cgpr.diagnostics.selection import SelectionResult


@dataclass
class ReAuditResult:
    clean: dict[int, list[dict]]
    accepted: dict[int, list[dict]]
    rejected: dict[int, list[dict]]
    cluster_reports: dict[int, list[dict]]


class ClusterDisagreementReAuditor:
    def __init__(self, config: CapacityAuditConfig):
        self.config = config

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    def run(
        self,
        audit: AuditResult,
        selection: SelectionResult,
        feature_bank: FeatureBank,
        prototype_bank: PrototypeBank,
    ) -> ReAuditResult:
        try:
            from sklearn.cluster import MiniBatchKMeans
        except Exception as exc:
            print(f"[WARN] sklearn unavailable for re-audit: {exc}")
            return ReAuditResult(
                clean=audit.clean,
                accepted={class_idx: [] for class_idx in range(self.config.num_classes)},
                rejected={class_idx: [] for class_idx in range(self.config.num_classes)},
                cluster_reports={class_idx: [] for class_idx in range(self.config.num_classes)},
            )

        features_norm = self._normalize(feature_bank.features)
        prototypes_norm = self._normalize(prototype_bank.prototypes)

        clean = {class_idx: list(audit.clean.get(class_idx, [])) for class_idx in range(self.config.num_classes)}
        accepted = {class_idx: [] for class_idx in range(self.config.num_classes)}
        rejected = {class_idx: [] for class_idx in range(self.config.num_classes)}
        cluster_reports = {class_idx: [] for class_idx in range(self.config.num_classes)}

        for class_idx in range(self.config.num_classes):
            pool = self._build_pool(
                class_idx=class_idx,
                audit=audit,
                selection=selection,
            )

            if not pool:
                continue

            pool_indices = np.asarray([int(item["index"]) for item in pool], dtype=np.int64)

            if len(pool_indices) < 10:
                class_accepted, class_rejected = self._small_pool_check(
                    class_idx=class_idx,
                    pool=pool,
                    features_norm=features_norm,
                    prototypes_norm=prototypes_norm,
                    feature_bank=feature_bank,
                )

                accepted[class_idx].extend(class_accepted)
                rejected[class_idx].extend(class_rejected)
                clean[class_idx].extend(class_accepted)

                print(
                    f"[ReAudit] {CLASS_NAMES[class_idx]:<12} "
                    f"pool={len(pool):4d} accepted={len(class_accepted):4d} small-pool"
                )
                continue

            labels = self._cluster_pool(pool_indices, features_norm)
            class_accepted, class_rejected, reports = self._evaluate_subclusters(
                class_idx=class_idx,
                pool=pool,
                pool_indices=pool_indices,
                labels=labels,
                features_norm=features_norm,
                prototypes_norm=prototypes_norm,
                feature_bank=feature_bank,
            )

            class_accepted = self._deduplicate_against_clean(
                class_idx=class_idx,
                clean=clean,
                candidates=class_accepted,
            )

            accepted[class_idx].extend(class_accepted)
            rejected[class_idx].extend(class_rejected)
            cluster_reports[class_idx].extend(reports)
            clean[class_idx].extend(class_accepted)

            print(
                f"[ReAudit] {CLASS_NAMES[class_idx]:<12} "
                f"pool={len(pool):4d} accepted={len(class_accepted):4d}"
            )

        return ReAuditResult(
            clean=clean,
            accepted=accepted,
            rejected=rejected,
            cluster_reports=cluster_reports,
        )

    def _build_pool(
        self,
        class_idx: int,
        audit: AuditResult,
        selection: SelectionResult,
    ) -> list[dict]:
        pool = []

        for item in audit.reaudit_pool.get(class_idx, []):
            record = dict(item)
            record["source_pool"] = "reaudit_pool"
            pool.append(record)

        for item in selection.reserve.get(class_idx, []):
            record = dict(item)
            record["source_pool"] = "reserve"
            pool.append(record)

        return pool

    def _cluster_pool(self, pool_indices: np.ndarray, features_norm: np.ndarray) -> np.ndarray:
        k = min(3, max(2, len(pool_indices) // 20))

        if len(pool_indices) < k:
            return np.zeros(len(pool_indices), dtype=np.int64)

        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=self.config.seed,
            batch_size=min(1024, max(32, len(pool_indices))),
            n_init="auto",
        )

        return kmeans.fit_predict(features_norm[pool_indices]).astype(np.int64)

    def _small_pool_check(
        self,
        class_idx: int,
        pool: list[dict],
        features_norm: np.ndarray,
        prototypes_norm: np.ndarray,
        feature_bank: FeatureBank,
    ) -> tuple[list[dict], list[dict]]:
        accepted = []
        rejected = []

        for item in pool:
            sample_idx = int(item["index"])
            metrics = self._sample_metrics(
                sample_idx=sample_idx,
                class_idx=class_idx,
                features_norm=features_norm,
                prototypes_norm=prototypes_norm,
                feature_bank=feature_bank,
            )

            ok = (
                metrics["class_prob"] >= 0.45
                and metrics["proto_sim"] >= 0.15
                and metrics["margin"] >= 0.03
            )

            record = dict(item)
            record.update(metrics)

            if ok:
                record["reaudit_accept_reason"] = "small_pool_individual_check"
                accepted.append(record)
            else:
                record["reaudit_reject_reason"] = "small_pool_failed_individual_check"
                rejected.append(record)

        accepted = sorted(
            accepted,
            key=lambda item: (
                float(item.get("class_prob", 0.0)),
                float(item.get("proto_sim", 0.0)),
                float(item.get("margin", 0.0)),
            ),
            reverse=True,
        )[: self.config.capacity_per_class]

        return accepted, rejected

    def _evaluate_subclusters(
        self,
        class_idx: int,
        pool: list[dict],
        pool_indices: np.ndarray,
        labels: np.ndarray,
        features_norm: np.ndarray,
        prototypes_norm: np.ndarray,
        feature_bank: FeatureBank,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        accepted = []
        rejected = []
        reports = []

        for cluster_id in sorted(set(labels.tolist())):
            mask = labels == cluster_id
            cluster_items = [pool[i] for i in np.where(mask)[0]]
            cluster_indices = pool_indices[mask]

            report = self._subcluster_report(
                class_idx=class_idx,
                cluster_id=cluster_id,
                cluster_indices=cluster_indices,
                features_norm=features_norm,
                prototypes_norm=prototypes_norm,
                feature_bank=feature_bank,
            )

            reports.append(report)

            if report["accepted"]:
                for item in cluster_items:
                    sample_idx = int(item["index"])
                    metrics = self._sample_metrics(
                        sample_idx=sample_idx,
                        class_idx=class_idx,
                        features_norm=features_norm,
                        prototypes_norm=prototypes_norm,
                        feature_bank=feature_bank,
                    )

                    if metrics["class_prob"] < 0.25:
                        rejected.append(self._reject_item(item, metrics, "low_class_prob_inside_accepted_subcluster"))
                        continue

                    if metrics["proto_sim"] < 0.05:
                        rejected.append(self._reject_item(item, metrics, "low_proto_sim_inside_accepted_subcluster"))
                        continue

                    record = dict(item)
                    record.update(metrics)
                    record["reaudit_accept_reason"] = "accepted_subcluster"
                    record["subcluster_id"] = int(cluster_id)
                    record["subcluster_score"] = float(report["subcluster_score"])
                    accepted.append(record)
            else:
                for item in cluster_items:
                    sample_idx = int(item["index"])
                    metrics = self._sample_metrics(
                        sample_idx=sample_idx,
                        class_idx=class_idx,
                        features_norm=features_norm,
                        prototypes_norm=prototypes_norm,
                        feature_bank=feature_bank,
                    )
                    rejected.append(self._reject_item(item, metrics, "rejected_subcluster"))

        accepted = sorted(
            accepted,
            key=lambda item: (
                float(item.get("subcluster_score", 0.0)),
                float(item.get("class_prob", 0.0)),
                float(item.get("proto_sim", 0.0)),
                float(item.get("margin", 0.0)),
            ),
            reverse=True,
        )[: self.config.capacity_per_class]

        return accepted, rejected, reports

    def _subcluster_report(
        self,
        class_idx: int,
        cluster_id: int,
        cluster_indices: np.ndarray,
        features_norm: np.ndarray,
        prototypes_norm: np.ndarray,
        feature_bank: FeatureBank,
    ) -> dict:
        class_probs = feature_bank.probs[cluster_indices, class_idx]

        sorted_probs = np.sort(feature_bank.probs[cluster_indices], axis=1)
        top1 = sorted_probs[:, -1]
        top2 = sorted_probs[:, -2]
        margins = top1 - top2

        proto_sims = features_norm[cluster_indices] @ prototypes_norm[class_idx]
        entropies = feature_bank.entropy[cluster_indices]

        rival_probs = feature_bank.probs[cluster_indices].copy()
        rival_probs[:, class_idx] = -1.0
        max_rival_probs = rival_probs.max(axis=1)

        mean_class_prob = float(class_probs.mean())
        mean_margin = float(margins.mean())
        mean_proto_sim = float(proto_sims.mean())
        mean_entropy = float(entropies.mean())
        mean_rival_prob = float(max_rival_probs.mean())

        subcluster_score = (
            0.40 * mean_class_prob
            + 0.30 * mean_proto_sim
            + 0.20 * mean_margin
            - 0.10 * mean_entropy
            - 0.15 * mean_rival_prob
        )

        accepted = (
            mean_class_prob >= 0.35
            and mean_proto_sim >= 0.10
            and mean_rival_prob <= 0.75
            and subcluster_score >= 0.05
        )

        return {
            "cluster_id": int(cluster_id),
            "size": int(len(cluster_indices)),
            "accepted": bool(accepted),
            "subcluster_score": float(subcluster_score),
            "mean_class_prob": mean_class_prob,
            "mean_margin": mean_margin,
            "mean_proto_sim": mean_proto_sim,
            "mean_entropy": mean_entropy,
            "mean_rival_prob": mean_rival_prob,
        }

    def _sample_metrics(
        self,
        sample_idx: int,
        class_idx: int,
        features_norm: np.ndarray,
        prototypes_norm: np.ndarray,
        feature_bank: FeatureBank,
    ) -> dict:
        sorted_probs = np.sort(feature_bank.probs[sample_idx])[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1])

        class_prob = float(feature_bank.probs[sample_idx, class_idx])
        proto_sim = float(features_norm[sample_idx] @ prototypes_norm[class_idx])
        entropy = float(feature_bank.entropy[sample_idx])

        rival_probs = feature_bank.probs[sample_idx].copy()
        rival_probs[class_idx] = -1.0
        max_rival_prob = float(rival_probs.max())

        return {
            "class_prob": class_prob,
            "proto_sim": proto_sim,
            "margin": margin,
            "entropy": entropy,
            "max_rival_prob": max_rival_prob,
        }

    @staticmethod
    def _reject_item(item: dict, metrics: dict, reason: str) -> dict:
        record = dict(item)
        record.update(metrics)
        record["reaudit_reject_reason"] = reason
        return record

    def _deduplicate_against_clean(
        self,
        class_idx: int,
        clean: dict[int, list[dict]],
        candidates: list[dict],
    ) -> list[dict]:
        clean_indices = {int(item["index"]) for item in clean.get(class_idx, [])}
        seen = set()
        output = []

        for item in candidates:
            sample_idx = int(item["index"])

            if sample_idx in clean_indices or sample_idx in seen:
                continue

            seen.add(sample_idx)
            output.append(item)

            if len(output) >= self.config.capacity_per_class:
                break

        return output
