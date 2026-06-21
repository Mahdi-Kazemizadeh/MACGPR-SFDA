from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    """Loads YAML configuration files."""

    @staticmethod
    def load(config_path: str | Path) -> dict[str, Any]:
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if config is None:
            raise ValueError(f"Config file is empty: {config_path}")

        return config
