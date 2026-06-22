import json
from pathlib import Path
from typing import Any

import numpy as np
from torchvision.datasets import ImageFolder


class MLLMQueryExporter:
    """Exports hard target samples as JSONL records for MLLM verification."""

    def __init__(
        self,
        target_dataset: ImageFolder,
        class_names: list[str],
        output_path: str | Path,
    ) -> None:
        self.target_dataset = target_dataset
        self.class_names = class_names
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        hard_mask: np.ndarray,
        pseudo_labels: np.ndarray,
        reliability_scores: np.ndarray,
    ) -> dict[str, Any]:
        hard_indices = np.where(hard_mask)[0]

        records = []

        with open(self.output_path, "w", encoding="utf-8") as file:
            for index in hard_indices:
                image_path, _ = self.target_dataset.samples[int(index)]
                label_index = int(pseudo_labels[int(index)])

                record = {
                    "sample_id": f"target_{int(index):08d}",
                    "target_index": int(index),
                    "image_path": str(image_path),
                    "model_label": self.class_names[label_index],
                    "model_label_index": label_index,
                    "reliability_score": float(reliability_scores[int(index)]),
                    "query_status": "pending",
                }

                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)

        return {
            "output_path": str(self.output_path),
            "num_queries": len(records),
        }
