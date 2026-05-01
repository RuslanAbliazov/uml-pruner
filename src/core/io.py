"""I/O utilities for loading and saving UML diagrams."""

import json
from pathlib import Path
from typing import Any


def load_diagram(path: str | Path) -> dict[str, Any]:
    """Load a UML diagram JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Dict with 'nodes' and 'edges' keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Diagram file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "nodes" not in data or "edges" not in data:
        raise ValueError(
            f"Invalid diagram format: missing 'nodes' or 'edges' in {path}"
        )

    return data


def save_diagram(data: dict[str, Any], path: str | Path) -> None:
    """Save a UML diagram to a JSON file.

    Args:
        data: Diagram data to save.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """Load a generic JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    """Save data to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
