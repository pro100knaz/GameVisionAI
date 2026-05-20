"""YAML/JSON configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON config file.

    Args:
        path: Path to config file (.yaml, .yml, or .json).

    Returns:
        Parsed config as a dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")

    if path.suffix == ".json":
        import json

        return json.loads(text)

    return yaml.safe_load(text) or {}
