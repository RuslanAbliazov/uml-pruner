"""Tiny prompt-template loader.

Each approach owns its own prompts under
``src/approaches/<name>/prompts/*.txt``. This module exposes a single
helper to read and format those files; it knows nothing about which
approach uses which template.

Usage
-----

>>> from pathlib import Path
>>> from src.llm.prompt_loader import load_prompt, render_prompt
>>>
>>> PROMPTS = Path(__file__).parent / "prompts"
>>> system = load_prompt(PROMPTS / "select_system.txt")
>>> user   = render_prompt(PROMPTS / "select_user.txt", query=q, candidates_json=js)

Templates use Python ``str.format()`` syntax. Literal braces must be
escaped as ``{{`` / ``}}``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=64)
def _read(path_str: str) -> str:
    """Read a prompt file, stripping a single trailing newline."""
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    text = path.read_text(encoding="utf-8")
    if text.endswith("\n"):
        text = text[:-1]
    return text


def load_prompt(path: str | Path) -> str:
    """Return the verbatim contents of a prompt template (cached)."""
    return _read(str(Path(path).resolve()))


def render_prompt(path: str | Path, **vars: Any) -> str:
    """Load a prompt template and substitute ``{var}`` placeholders."""
    template = load_prompt(path)
    try:
        return template.format(**vars)
    except KeyError as e:
        raise KeyError(
            f"Missing variable {e} for prompt '{path}'. "
            f"Provided keys: {sorted(vars)}"
        ) from None


def clear_cache() -> None:
    """Drop cached templates (useful in tests after editing files on disk)."""
    _read.cache_clear()


__all__ = ["load_prompt", "render_prompt", "clear_cache"]
