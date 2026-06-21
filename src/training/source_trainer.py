import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.io import ExperimentIO
from src.utils.metrics import ClassificationMetrics


class SourceTrainer:
    """Trains a source-domain classifier for source-free domain adaptation."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.training_config = config["training"]
        self.output_config = config["output"]
        self.model_config = config["model"]
        self.dataset_config = config["dataset"]

        self.best_val_accuracy = -1.0
        self.best_state_dict = None
        self.history: list[dict[str, Any]] = []

    def train(self) -> dict[str, Any]:
        self.model.to(self.device)

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.training_config["lr"],
            momentum=self.training_config["momentum"],
            weight_decay=self.training_config["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.training_config["epochs"],
        )

        patience = int(self.training_config["early_stop_patience"])
        patience_counter = 0

        total_start_time = time.perf_counter()

        for epoch in range(1, self.training_config["epochs"] + 1):
            epoch_start_time = time.perf_counter()

            train_loss = self._train_one_epoch(optimizer)
            val_metrics = self.validate()

            scheduler.step()

            epoch_time = time.perf_counter() - epoch_start_time

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_mean_class_accuracy": val_metrics["mean_class_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_seconds": round(epoch_time, 3),
            }

            self.history.append(epoch_record)

            print(
                f"Epoch {epoch}/{self.training_config['epochs']} | "
                f"Loss={train_loss:.4f} | "
                f"ValAcc={val_metrics['accuracy']:.4f} | "
                f"MCA={val_metrics['mean_class_accuracy']:.4f} | "
                f"MacroF1={val_metrics['macro_f1']:.4f} | "
                f"Time={ExperimentIO.format_seconds(epoch_time)}"
            )

            if val_metrics["accuracy"] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics["accuracy"]
                self.best_state_dict = deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        total_time = time.perf_counter() - total_start_time

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        checkpoint_path = self.save_checkpoint(total_time)

        return {
            "best_val_accuracy": self.best_val_accuracy,
            "checkpoint_path": str(checkpoint_path),
            "total_time_seconds": round(total_time, 3),
            "history": self.history,
        }

    def _train_one_epoch(self, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / max(total_samples, 1)

    @torch.no_grad()
    def validate(self) -> dict[str, Any]:
        self.model.eval()

        predictions = []
        targets = []

        for images, labels in self.val_loader:
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

    def save_checkpoint(self, total_time: float) -> Path:
        checkpoint_dir = ExperimentIO.ensure_dir(
            self.output_config["checkpoint_dir"])
        checkpoint_path = checkpoint_dir / \
            self.output_config["checkpoint_name"]

        checkpoint = {
            "model_state": self.model.state_dict(),
            "best_val_accuracy": self.best_val_accuracy,
            "history": self.history,
            "config": self.config,
            "backbone": self.model_config["backbone"],
            "num_classes": self.dataset_config["num_classes"],
            "class_names": self.dataset_config["class_names"],
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "total_time_seconds": round(total_time, 3),
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved source checkpoint to: {checkpoint_path}")

        return checkpoint_path
