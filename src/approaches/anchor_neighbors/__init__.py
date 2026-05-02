"""Подход #2: anchor + neighbors + prune.

Конвейер из 4 этапов, каждый этап — отдельный модуль. Состав пакета:

    settings.py                — чтение конфига, build_runner (фабрика)
    pipeline.py                — оркестратор всех 4 этапов
    stage_outputs.py           — общий контракт `StageOutcome`
    stage1_retrieve.py         — RAG: top-K кандидатов
    stage2_select_anchor.py    — LLM: выбор anchor
    stage3_expand_neighbors.py — 1-hop соседи
    stage4_prune.py            — LLM: required/useful/irrelevant
    prompt_templates.py        — обёртки над ./prompts/*.txt
    ground_truth.py            — загрузка эталона по (repo, query)
    metrics.py                 — оценка качества каждого этапа
    debug_report.py            — JSONL-отчёт + агрегация
    run.py                     — CLI

Локальный CLI:

    python src/approaches/anchor_neighbors/run.py --until rag       # только RAG
    python src/approaches/anchor_neighbors/run.py --until prune     # полный прогон

`build_runner` импортируется реестром подходов (`src/approaches/__init__.py`).
"""

from src.approaches.anchor_neighbors.pipeline import NAME
from src.approaches.anchor_neighbors.settings import build_runner

__all__ = ["NAME", "build_runner"]
