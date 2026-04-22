"""Prompt templates loaded from .txt files.

Templates live in the `prompts/` directory at the project root (or any
directory configured via `set_prompts_dir`). They are read from disk on first
use and cached in memory.

Template variables use Python .format() syntax. Literal braces in the templates
are written as `{{` / `}}` (standard Python escape).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Template directory resolution
# -----------------------------------------------------------------------------

_DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
_prompts_dir: Path = _DEFAULT_PROMPTS_DIR


def set_prompts_dir(path: str | Path) -> None:
    """Override the directory where prompt .txt files are looked up."""
    global _prompts_dir
    _prompts_dir = Path(path).resolve()
    _read_template.cache_clear()
    logger.info("Prompts directory set to %s", _prompts_dir)


def get_prompts_dir() -> Path:
    return _prompts_dir


@lru_cache(maxsize=32)
def _read_template(filename: str) -> str:
    """Read a prompt template file, stripping trailing newline for cleanliness."""
    path = _prompts_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        content = f.read()
    # Remove exactly one trailing newline (common for text files) but keep
    # internal blank lines intact.
    if content.endswith("\n"):
        content = content[:-1]
    return content


def _render(filename: str, **kwargs: Any) -> str:
    """Load template and format it with the given keyword arguments."""
    template = _read_template(filename)
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise KeyError(
            f"Missing variable {e} for prompt template '{filename}'. "
            f"Provided keys: {list(kwargs.keys())}"
        ) from None


# -----------------------------------------------------------------------------
# Stage 1 — Package-level coarse filtering
# -----------------------------------------------------------------------------


def stage1_system_prompt() -> str:
    return _read_template("stage1_system.txt")


def build_stage1_user_prompt(
    query: str,
    packages: list[dict[str, Any]],
    batch_idx: int,
    total_batches: int,
) -> str:
    package_lines = []
    for i, pkg in enumerate(packages, 1):
        samples = ", ".join(pkg.get("samples", [])[:8])
        package_lines.append(f"{i}. {pkg['name']} ({pkg['count']} classes) — {samples}")
    packages_text = "\n".join(package_lines)
    return _render(
        "stage1_user.txt",
        query=query,
        batch_idx=batch_idx,
        total_batches=total_batches,
        packages_text=packages_text,
    )


# -----------------------------------------------------------------------------
# Stage 2 — Class-level refinement
# -----------------------------------------------------------------------------


def stage2_system_prompt() -> str:
    return _read_template("stage2_system.txt")


def build_stage2_user_prompt(
    query: str,
    classes: list[dict[str, Any]],
    batch_idx: int,
    total_batches: int,
) -> str:
    classes_json = json.dumps(classes, ensure_ascii=False)
    return _render(
        "stage2_user.txt",
        query=query,
        batch_idx=batch_idx,
        total_batches=total_batches,
        classes_json=classes_json,
    )


# -----------------------------------------------------------------------------
# Stage 3 — Final pruning
# -----------------------------------------------------------------------------


def stage3_system_prompt() -> str:
    return _read_template("stage3_system.txt")


def build_stage3_user_prompt(
    query: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    target_min: int = 10,
    target_max: int = 50,
) -> str:
    subgraph_json = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)
    return _render(
        "stage3_user.txt",
        query=query,
        subgraph_json=subgraph_json,
        target_min=target_min,
        target_max=target_max,
    )


# -----------------------------------------------------------------------------
# Backwards-compatibility constants (lazy via module __getattr__)
# -----------------------------------------------------------------------------
# Stages currently import STAGE{1,2,3}_SYSTEM_PROMPT as module-level constants.
# We keep these names but resolve them lazily so edits to .txt files are picked
# up after a cache clear.


def __getattr__(name: str) -> Any:
    if name == "STAGE1_SYSTEM_PROMPT":
        return stage1_system_prompt()
    if name == "STAGE2_SYSTEM_PROMPT":
        return stage2_system_prompt()
    if name == "STAGE3_SYSTEM_PROMPT":
        return stage3_system_prompt()
    raise AttributeError(f"module 'src.llm.prompts' has no attribute {name!r}")
