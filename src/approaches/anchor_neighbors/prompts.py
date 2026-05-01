"""Thin wrappers over the .txt prompt files in ``./prompts/``.

Two LLM stages, each with its own pair of system + user templates:
``select_*`` for anchor selection and ``prune_*`` for the final pruning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.llm.prompt_loader import load_prompt, render_prompt

_DIR = Path(__file__).parent / "prompts"


def select_system() -> str:
    return load_prompt(_DIR / "select_system.txt")


def select_user(query: str, candidates: list[dict[str, Any]]) -> str:
    return render_prompt(
        _DIR / "select_user.txt",
        query=query,
        candidates_json=json.dumps(candidates, ensure_ascii=False, indent=2),
    )


def prune_system() -> str:
    return load_prompt(_DIR / "prune_system.txt")


def prune_user(
    query: str,
    anchor: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> str:
    return render_prompt(
        _DIR / "prune_user.txt",
        query=query,
        anchor=anchor,
        subgraph_json=json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False),
    )
