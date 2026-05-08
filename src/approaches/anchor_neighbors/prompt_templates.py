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


def prune_step_system(step_name: str) -> str:
    """Загрузить system промпт для конкретного шага прунинга.
    
    Для шага "single" использует prune_system.txt (обратная совместимость).
    Для других шагов ищет prune_{step_name}_system.txt.
    """
    if step_name == "single":
        return load_prompt(_PROMPTS_DIR / "prune_system.txt")
    return load_prompt(_PROMPTS_DIR / f"prune_{step_name}_system.txt")


def prune_step_user(
    step_name: str,
    query: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> str:
    """Сгенерировать user промпт для конкретного шага прунинга.
    
    Для шага "single" использует prune_user.txt (обратная совместимость).
    Для других шагов ищет prune_{step_name}_user.txt и передает в него:
    - query
    - subgraph_json
    - context (результаты предыдущих шагов)
    """
    subgraph_json = json.dumps(
        {"nodes": nodes, "edges": edges}, ensure_ascii=False
    )
    
    if step_name == "single":
        return render_prompt(
            _PROMPTS_DIR / "prune_user.txt",
            query=query,
            subgraph_json=subgraph_json,
        )
    
    # Подготовим переменные для шаблона
    template_vars = {
        "query": query,
        "subgraph_json": subgraph_json,
    }
    
    # Добавим context, если есть
    if context:
        for key, value in context.items():
            if isinstance(value, (list, dict)):
                template_vars[key] = json.dumps(value, ensure_ascii=False, indent=2)
            else:
                template_vars[key] = str(value)
    
    return render_prompt(
        _PROMPTS_DIR / f"prune_{step_name}_user.txt",
        **template_vars,
    )
