"""Config loading utilities.

Supports ${ENV_VAR} substitution in string values: any value of the form
"${NAME}" is replaced with os.environ["NAME"] at load time (returns empty
string if the variable is not set). Useful for keeping secrets out of the
YAML file itself.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_VAR_PATTERN = re.compile(r"^\$\{([A-Z_][A-Z0-9_]*)\}$")


def _resolve_env_vars(value: Any) -> Any:
    """Recursively replace ${ENV_VAR} references in a loaded YAML tree."""
    if isinstance(value, str):
        match = _ENV_VAR_PATTERN.match(value.strip())
        if match:
            return os.environ.get(match.group(1), "")
        return value
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


class Config:
    """Simple attribute-access wrapper over nested YAML config."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        if key not in self._data:
            raise AttributeError(f"Config has no attribute '{key}'")
        value = self._data[key]
        if isinstance(value, dict):
            return Config(value)
        return value

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        if isinstance(value, dict):
            return Config(value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        value = self._data.get(key, default)
        if isinstance(value, dict):
            return Config(value)
        return value

    def to_dict(self) -> dict[str, Any]:
        return self._data


def load_config(path: str | Path = "configs/config.yaml") -> Config:
    """Load a YAML config file, resolve ${ENV_VAR} references, and return
    a Config wrapper.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data = _resolve_env_vars(data)
    return Config(data)
