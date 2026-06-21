from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.utils.io import ExperimentIO
from src.utils.metrics import ClassificationMetrics


class SourceEvaluator:
    """Evaluates a source-trained model on a target domain."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_loader: torch.utils.data.DataLoader,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model
        self.target_loader = target_loader
        self.config = config
        self.device = device

        self.dataset_config = config["dataset"]
        self.output_config = config["output"]

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        self.model.to(self.device)
        self.model.eval()

        predictions = []
        targets = []

        for images, labels in self.target_loader:
            images = images.to(self.device)

            logits = self.model(images)
            batch_predictions = logits.argmax(dim=1).cpu().numpy()

            predictions.append(batch_predictions)
            targets.append(labels.numpy())

        y_pred = np.concatenate(predictions, axis=0)
        y_true = np.concatenate(targets, axis=0)

        return ClassificationMetrics.compute(
            y_true=y_true,
            y_pred=y_pred,
            num_classes=int(self.dataset_config["num_classes"]),
        )

    def save_metrics(self, metrics: dict[str, Any]) -> Path:
        results_dir = ExperimentIO.ensure_dir(self.output_config["results_dir"])
        metrics_path = results_dir / "metrics.json"

        ExperimentIO.save_json(metrics, metrics_path)

        return metrics_path
