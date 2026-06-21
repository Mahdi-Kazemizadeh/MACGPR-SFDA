from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class ClassificationMetrics:
    """Computes classification metrics for adaptation experiments."""

    @staticmethod
    def mean_class_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_classes: int,
    ) -> tuple[float, list[float]]:
        per_class_accuracy = []

        for class_index in range(num_classes):
            class_mask = y_true == class_index

            if class_mask.sum() == 0:
                per_class_accuracy.append(0.0)
                continue

            correct = (y_pred[class_mask] == class_index).sum()
            total = class_mask.sum()
            per_class_accuracy.append(float(correct / total))

        return float(np.mean(per_class_accuracy)), per_class_accuracy

    @staticmethod
    def compute(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_classes: int,
    ) -> dict[str, Any]:
        accuracy = float(accuracy_score(y_true, y_pred))
        mean_class_acc, per_class_acc = ClassificationMetrics.mean_class_accuracy(
            y_true=y_true,
            y_pred=y_pred,
            num_classes=num_classes,
        )

        macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
        per_class_f1 = f1_score(
            y_true,
            y_pred,
            average=None,
            labels=list(range(num_classes)),
            zero_division=0,
        ).tolist()

        return {
            "accuracy": accuracy,
            "mean_class_accuracy": mean_class_acc,
            "macro_f1": macro_f1,
            "per_class_accuracy": per_class_acc,
            "per_class_f1": per_class_f1,
        }
