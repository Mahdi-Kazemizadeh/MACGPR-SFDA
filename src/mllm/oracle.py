from abc import ABC, abstractmethod
from pathlib import Path

from src.mllm.cache import MLLMCache, MLLMResponse


class BaseMLLMOracle(ABC):
    """Base interface for MLLM-based label verification."""

    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names

    @abstractmethod
    def query(
        self,
        sample_id: str,
        image_path: str | Path,
        candidate_labels: list[str] | None = None,
    ) -> MLLMResponse | None:
        """Return an MLLM response for one image."""
        raise NotImplementedError


class CachedMLLMOracle(BaseMLLMOracle):
    """Reads MLLM responses from cache without calling any external API."""

    def __init__(
        self,
        class_names: list[str],
        cache_path: str | Path,
        source_name: str = "cache",
    ) -> None:
        super().__init__(class_names=class_names)
        self.cache = MLLMCache(cache_path=cache_path)
        self.source_name = source_name

    def query(
        self,
        sample_id: str,
        image_path: str | Path,
        candidate_labels: list[str] | None = None,
    ) -> MLLMResponse | None:
        cached_response = self.cache.get(sample_id)

        if cached_response is None:
            return None

        if not self._is_valid_label(cached_response.predicted_label):
            return MLLMResponse(
                sample_id=sample_id,
                image_path=str(image_path),
                predicted_label=cached_response.predicted_label,
                confidence=0.0,
                is_valid=False,
                source=self.source_name,
                explanation="Cached label is not part of the configured class list.",
            )

        return cached_response

    def save_response(self, response: MLLMResponse) -> None:
        if not self._is_valid_label(response.predicted_label):
            response.is_valid = False
            response.confidence = 0.0

        self.cache.set(response)

    def _is_valid_label(self, label: str) -> bool:
        return label in self.class_names
