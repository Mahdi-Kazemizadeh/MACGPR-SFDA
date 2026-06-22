import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class MLLMResponse:
    """Stores one cached MLLM verification response."""

    sample_id: str
    image_path: str
    predicted_label: str
    confidence: float
    is_valid: bool
    source: str
    explanation: str | None = None


class MLLMCache:
    """JSONL cache for MLLM responses.

    This cache avoids repeated MLLM calls for the same target sample.
    """

    def __init__(self, cache_path: str | Path) -> None:
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, MLLMResponse] = {}
        self._load()

    def _load(self) -> None:
        if not self.cache_path.exists():
            return

        with open(self.cache_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                response = MLLMResponse(**data)
                self._items[response.sample_id] = response

    def get(self, sample_id: str) -> MLLMResponse | None:
        return self._items.get(sample_id)

    def set(self, response: MLLMResponse) -> None:
        self._items[response.sample_id] = response
        self._rewrite()

    def contains(self, sample_id: str) -> bool:
        return sample_id in self._items

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_path": str(self.cache_path),
            "num_items": len(self._items),
        }

    def _rewrite(self) -> None:
        with open(self.cache_path, "w", encoding="utf-8") as file:
            for response in self._items.values():
                file.write(json.dumps(asdict(response),
                           ensure_ascii=False) + "\n")
