import json
from pathlib import Path
from typing import Any


class ExperimentIO:
    """Handles basic experiment input/output utilities."""

    @staticmethod
    def ensure_dir(path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def save_json(data: dict[str, Any], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)

    @staticmethod
    def load_json(path: str | Path) -> dict[str, Any]:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def format_seconds(seconds: float) -> str:
        minutes = int(seconds) // 60
        remaining_seconds = seconds - minutes * 60

        if minutes > 0:
            return f"{minutes}m {remaining_seconds:.1f}s"

        return f"{remaining_seconds:.1f}s"
