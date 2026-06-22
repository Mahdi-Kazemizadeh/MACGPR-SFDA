import json
from pathlib import Path
from typing import Any

import numpy as np


class MLLMResponseIngestion:
    """Loads cached MLLM responses and converts them into training labels.

    This module does not call any external MLLM API.
    It only consumes previously generated/cache responses.
    """

    def __init__(
        self,
        class_names: list[str],
        cache_path: str | Path,
        min_confidence: float = 0.70,
    ) -> None:
        self.class_names = class_names
        self.class_to_index = {
            class_name: index for index, class_name in enumerate(class_names)
        }
        self.cache_path = Path(cache_path)
        self.min_confidence = float(min_confidence)

    def load_verified_labels(
        self,
        target_size: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return verified mask and labels from cached MLLM responses.

        verified_mask[index] is True only when MLLM response is valid and confident.
        verified_labels[index] stores the corresponding class index.
        """
        verified_mask = np.zeros(target_size, dtype=bool)
        verified_labels = np.full(target_size, fill_value=-1, dtype=np.int64)

        stats = {
            "cache_path": str(self.cache_path),
            "num_cache_records": 0,
            "num_valid_records": 0,
            "num_used_records": 0,
            "num_rejected_low_confidence": 0,
            "num_rejected_invalid_label": 0,
            "num_rejected_invalid_status": 0,
        }

        if not self.cache_path.exists():
            stats["cache_exists"] = False
            return verified_mask, verified_labels, stats

        stats["cache_exists"] = True

        with open(self.cache_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                stats["num_cache_records"] += 1
                record = json.loads(line)

                parsed = self._parse_record(
                    record=record, target_size=target_size)

                if parsed is None:
                    stats["num_rejected_invalid_status"] += 1
                    continue

                target_index, label_name, confidence = parsed

                if label_name not in self.class_to_index:
                    stats["num_rejected_invalid_label"] += 1
                    continue

                stats["num_valid_records"] += 1

                if confidence < self.min_confidence:
                    stats["num_rejected_low_confidence"] += 1
                    continue

                verified_mask[target_index] = True
                verified_labels[target_index] = self.class_to_index[label_name]
                stats["num_used_records"] += 1

        return verified_mask, verified_labels, stats

    def _parse_record(
        self,
        record: dict[str, Any],
        target_size: int,
    ) -> tuple[int, str, float] | None:
        """Parse one MLLM response record.

        Expected fields:
        - target_index
        - predicted_label
        - confidence
        - is_valid
        """
        if not bool(record.get("is_valid", False)):
            return None

        if "target_index" not in record:
            return None

        target_index = int(record["target_index"])

        if target_index < 0 or target_index >= target_size:
            return None

        label_name = str(record.get("predicted_label", ""))
        confidence = float(record.get("confidence", 0.0))

        return target_index, label_name, confidence
