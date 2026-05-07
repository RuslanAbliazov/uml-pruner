"""Тонкие обёртки над текстовыми шаблонами в ./prompts/.

У нас два LLM-этапа, у каждого пара system+user шаблон:

    select_system / select_user   — выбор anchor (этап 2)
    prune_system  / prune_user    — прунинг подграфа (этап 4)

Этот модуль ИСКЛЮЧИТЕЛЬНО склеивает шаблоны с переменными. Никакой логики
самого пайплайна тут быть не должно — это даёт возможность редактировать
промпты в .txt без правки Python.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.llm.prompt_loader import load_prompt, render_prompt

_PROMPTS_DIR = Path(__file__).parent / "prompts"


# ---- этап 2: выбор anchor ------------------------------------------------

def anchor_selection_system() -> str:
    return load_prompt(_PROMPTS_DIR / "select_system.txt")


def anchor_selection_user(query: str, candidates: list[dict[str, Any]]) -> str:
    """Заметь: ``candidates_json`` подаётся уже отформатированной строкой,
    а не объектом — так LLM видит читаемый отступ-2 JSON, а сам шаблон
    остаётся однострочным с одним placeholder'ом."""
    return render_prompt(
        _PROMPTS_DIR / "select_user.txt",
        query=query,
        candidates_json=json.dumps(candidates, ensure_ascii=False, indent=2),
    )


# ---- этап 4: прунинг подграфа -------------------------------------------

def prune_system() -> str:
    return load_prompt(_PROMPTS_DIR / "prune_system.txt")


def prune_user(
    query: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> str:
    return render_prompt(
        _PROMPTS_DIR / "prune_user.txt",
        query=query,
        subgraph_json=json.dumps(
            {"nodes": nodes, "edges": edges}, ensure_ascii=False
        ),
    )
