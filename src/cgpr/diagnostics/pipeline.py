from __future__ import annotations

import random

import numpy as np
import torch


from cgpr.diagnostics.gap_analysis import GapAnalyzer
from cgpr.diagnostics.audit import AuditEngine
from cgpr.diagnostics.clustering import ClusterAnalyzer
from cgpr.diagnostics.config import CapacityAuditConfig
from cgpr.diagnostics.data import VisDATargetDataModule
from cgpr.diagnostics.feature_bank import TargetFeatureExtractor
from cgpr.diagnostics.model import SourceModelLoader
from cgpr.diagnostics.prototype import PrototypeBuilder
from cgpr.diagnostics.reaudit import ClusterDisagreementReAuditor
from cgpr.diagnostics.replacement import ReplacementEngine
from cgpr.diagnostics.report import ReportBuilder
from cgpr.diagnostics.scoring import ScoreComputer
from cgpr.diagnostics.selection import CapacitySelector


class CapacityAuditPipeline:
    def __init__(self, config: CapacityAuditConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> dict:
        self._set_seed()

        self._print_header()

        data_module = VisDATargetDataModule(
            visda_root=self.config.visda_root,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )
        target_data = data_module.build()

        model = SourceModelLoader(
            num_classes=self.config.num_classes,
            checkpoint_path=self.config.checkpoint,
            device=self.device,
        ).build()

        feature_bank = TargetFeatureExtractor(
            model=model,
            device=self.device,
        ).extract(target_data.loader)

        prototype_bank = PrototypeBuilder(
            seed_per_class=self.config.prototype_seed_per_class,
        ).build(feature_bank)

        clustering = ClusterAnalyzer(
            num_classes=self.config.num_classes,
            seed=self.config.seed,
        ).run(feature_bank)

        scores = ScoreComputer().compute(
            feature_bank=feature_bank,
            prototype_bank=prototype_bank,
            clustering=clustering,
        )

        selection = CapacitySelector(
            capacity_per_class=self.config.capacity_per_class,
            reserve_per_class=self.config.reserve_per_class,
        ).select(scores)

        audit = AuditEngine(
            config=self.config,
        ).audit(
            selection=selection,
            feature_bank=feature_bank,
            clustering=clustering,
            scores=scores,
        )

        reaudit = ClusterDisagreementReAuditor(
            config=self.config,
        ).run(
            audit=audit,
            selection=selection,
            feature_bank=feature_bank,
            prototype_bank=prototype_bank,
        )

        GapAnalyzer(
            config=self.config,
        ).export(
            feature_bank=feature_bank,
            prototype_bank=prototype_bank,
            clustering=clustering,
            scores=scores,
            selection=selection,
            audit=audit,
            reaudit=reaudit,
        )

        replacement = ReplacementEngine(
            config=self.config,
        ).run(
            selection=selection,
            audit=audit,
            reaudit=reaudit,
            feature_bank=feature_bank,
            prototype_bank=prototype_bank,
            clustering=clustering,
        )

        reporter = ReportBuilder(config=self.config)

        report = reporter.build(
            feature_bank=feature_bank,
            prototype_bank=prototype_bank,
            clustering=clustering,
            scores=scores,
            selection=selection,
            audit=audit,
            reaudit=reaudit,
            replacement=replacement,
        )

        reporter.print_report(report)
        reporter.save(
            report=report,
            audit=audit,
            reaudit=reaudit,
            replacement=replacement,
        )

        return report

    def _set_seed(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)

    def _print_header(self) -> None:
        print("=" * 90)
        print("Capacity-Aware Audit Diagnostic | OOP Pipeline")
        print("=" * 90)
        print(f"Device: {self.device}")
        print(f"VISDA_ROOT: {self.config.visda_root}")
        print(f"Checkpoint: {self.config.checkpoint}")
        print(f"Capacity per class: {self.config.capacity_per_class}")
        print(f"Reserve per class: {self.config.reserve_per_class}")
