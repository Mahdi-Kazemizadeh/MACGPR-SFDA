import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from torchvision.datasets import ImageFolder


class MLLMQueryExporter:
    """Exports hard target samples for MLLM verification.

    Two exports are created:
    1. Raw export: keeps the original image path for internal analysis.
    2. Sanitized export: copies images to neutral filenames to avoid class-name leakage.
    """

    def __init__(
        self,
        target_dataset: ImageFolder,
        class_names: list[str],
        output_dir: str | Path,
    ) -> None:
        self.target_dataset = target_dataset
        self.class_names = class_names
        self.output_dir = Path(output_dir)

        self.raw_output_path = self.output_dir / "mllm_queries_raw.jsonl"
        self.sanitized_output_path = self.output_dir / "mllm_queries_sanitized.jsonl"
        self.sanitized_image_dir = self.output_dir / "mllm_images"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sanitized_image_dir.mkdir(parents=True, exist_ok=True)

        self.easy_raw_output_path = self.output_dir / "easy_samples_raw.jsonl"

    def export_easy_raw(
        self,
        easy_mask: np.ndarray,
        pseudo_labels: np.ndarray,
        reliability_scores: np.ndarray,
    ) -> dict[str, Any]:
        """Export easy samples with original paths for internal analysis only."""
        easy_indices = np.where(easy_mask)[0]

        with open(self.easy_raw_output_path, "w", encoding="utf-8") as file:
            for index in easy_indices:
                index = int(index)

                original_image_path, _ = self.target_dataset.samples[index]
                original_image_path = Path(original_image_path)

                label_index = int(pseudo_labels[index])
                sample_id = f"target_{index:08d}"
                reliability_score = float(reliability_scores[index])

                record = {
                    "sample_id": sample_id,
                    "target_index": index,
                    "image_path": str(original_image_path),
                    "model_label": self.class_names[label_index],
                    "model_label_index": label_index,
                    "reliability_score": reliability_score,
                    "selection_group": "easy",
                    "export_type": "easy_raw_internal_analysis",
                }

                file.write(json.dumps(record, ensure_ascii=False) + "\n")

        return {
            "easy_raw_output_path": str(self.easy_raw_output_path),
            "num_easy_samples": int(len(easy_indices)),
        }

    def export(
        self,
        query_mask: np.ndarray,
        pseudo_labels: np.ndarray,
        reliability_scores: np.ndarray,
    ) -> dict[str, Any]:
        hard_indices = np.where(query_mask)[0]

        with open(self.raw_output_path, "w", encoding="utf-8") as raw_file, open(
            self.sanitized_output_path, "w", encoding="utf-8"
        ) as sanitized_file:
            for index in hard_indices:
                index = int(index)

                original_image_path, _ = self.target_dataset.samples[index]
                original_image_path = Path(original_image_path)

                label_index = int(pseudo_labels[index])
                sample_id = f"target_{index:08d}"
                reliability_score = float(reliability_scores[index])

                raw_record = self._build_raw_record(
                    sample_id=sample_id,
                    target_index=index,
                    original_image_path=original_image_path,
                    label_index=label_index,
                    reliability_score=reliability_score,
                )

                sanitized_record = self._build_sanitized_record(
                    sample_id=sample_id,
                    target_index=index,
                    original_image_path=original_image_path,
                    label_index=label_index,
                    reliability_score=reliability_score,
                )

                raw_file.write(json.dumps(
                    raw_record, ensure_ascii=False) + "\n")
                sanitized_file.write(
                    json.dumps(sanitized_record, ensure_ascii=False) + "\n"
                )

        return {
            "raw_output_path": str(self.raw_output_path),
            "sanitized_output_path": str(self.sanitized_output_path),
            "sanitized_image_dir": str(self.sanitized_image_dir),
            "num_queries": int(len(hard_indices)),
        }

    def _build_raw_record(
        self,
        sample_id: str,
        target_index: int,
        original_image_path: Path,
        label_index: int,
        reliability_score: float,
    ) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "target_index": target_index,
            "image_path": str(original_image_path),
            "model_label": self.class_names[label_index],
            "model_label_index": label_index,
            "reliability_score": reliability_score,
            "query_status": "pending",
            "export_type": "raw_internal_analysis",
        }

    def _build_sanitized_record(
        self,
        sample_id: str,
        target_index: int,
        original_image_path: Path,
        label_index: int,
        reliability_score: float,
    ) -> dict[str, Any]:
        sanitized_image_path = self._copy_to_sanitized_path(
            sample_id=sample_id,
            original_image_path=original_image_path,
        )

        return {
            "sample_id": sample_id,
            "target_index": target_index,
            "image_path": str(sanitized_image_path),
            "model_label": self.class_names[label_index],
            "model_label_index": label_index,
            "reliability_score": reliability_score,
            "query_status": "pending",
            "export_type": "sanitized_mllm_query",
        }

    def _copy_to_sanitized_path(
        self,
        sample_id: str,
        original_image_path: Path,
    ) -> Path:
        suffix = original_image_path.suffix.lower()
        if suffix not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            suffix = ".jpg"

        sanitized_image_path = self.sanitized_image_dir / \
            f"{sample_id}{suffix}"

        if not sanitized_image_path.exists():
            shutil.copy2(original_image_path, sanitized_image_path)

        return sanitized_image_path
