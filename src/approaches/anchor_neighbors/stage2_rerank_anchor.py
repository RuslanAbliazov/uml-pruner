"""Этап 2 (альтернатива) — cross-encoder reranker выбирает anchor.

Берёт top-K кандидатов от RAG (этап 1), скорит каждый против запроса
кросс-энкодером и возвращает StageOutcome с node_id наивысшего score.

Контракт совместим с ``stage2_select_anchor.select_anchor``:

* На вход — те же поля ``query`` / ``candidates`` / ``node_by_id`` /
  ``edges``. ``llm`` / ``tracer`` / ``sample_id`` принимаются для
  единообразия, но не используются (LLM-вызовов здесь нет).
* На выход — `StageOutcome(stage=ANCHOR, node_ids=[chosen], payload={...})`
  с тем же набором ключей в ``payload``, что и LLM-вариант, плюс
  ``payload.scores`` — отсортированный по убыванию список ``[node_id, score]``
  для дебага.
* ``payload.fallback`` всегда ``False``: галлюцинаций тут не бывает,
  reranker физически не может вернуть id вне списка кандидатов.

Метрики (см. ``metrics.py``) и debug-отчёт работают без изменений: они
смотрят только на ``stage`` / ``node_ids`` / ``payload['fallback']`` и
``payload['anchor']``.

Тексты узлов формируются тем же ``src.rag.node_to_text``, что используется
для построения индекса эмбеддингов — это даёт ретриверу и реранкеру одно и
то же представление узла, без рассинхрона.
"""

from __future__ import annotations

import time
from typing import Any

from src.approaches.anchor_neighbors.stage_outputs import StageName, StageOutcome
from src.rag.node_to_text import EdgeIndex, node_to_text
from src.rag.reranker import LocalReranker


def _candidate_texts(
    candidates: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    edge_index: EdgeIndex,
) -> list[str]:
    """Собрать текстовое представление каждого кандидата для reranker'а.

    Если в диаграмме нет узла под node_id (рассинхрон индекса) — отдадим
    короткое имя как минимально валидный текст, чтобы reranker не падал.
    """
    texts: list[str] = []
    for c in candidates:
        node = node_by_id.get(c["node_id"])
        if node is None:
            texts.append(c.get("name") or c["node_id"])
            continue
        texts.append(node_to_text(node, edge_index=edge_index))
    return texts


async def select_anchor_via_reranker(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    edges: list[dict[str, str]],
    reranker: LocalReranker,
    # Принято для единообразия с LLM-вариантом; не используется.
    llm: Any = None,
    tracer: Any = None,
    sample_id: str = "",
) -> StageOutcome:
    """Выбрать anchor через cross-encoder reranker. Возвращает StageOutcome.

    `async`, чтобы сигнатура совпадала с LLM-вариантом и pipeline мог
    вызывать через ``await``. Сам reranker синхронный (CPU/GPU-bound), но
    обёртка от этого не страдает — выполнение быстрое и блокирующее именно
    то, что нам нужно (один прогон на сэмпл).
    """
    if not candidates:
        return StageOutcome(
            stage=StageName.ANCHOR,
            aborted="no_candidates",
            info={"reason": "stage 1 returned no candidates"},
        )

    # EdgeIndex строится на каждый сэмпл — это O(|edges|), невелико на фоне
    # вызова reranker'а. Зато код проще и не нужен общий кэш.
    edge_index = EdgeIndex.from_edges(edges)
    texts = _candidate_texts(candidates, node_by_id, edge_index)

    started = time.time()
    try:
        scores = reranker.score(query, texts)
    except Exception as e:  # noqa: BLE001 — единая граница внешних сбоев
        return StageOutcome(
            stage=StageName.ANCHOR,
            aborted="reranker_failed",
            info={"error": repr(e), "elapsed_s": round(time.time() - started, 2)},
        )

    # scores выровнен с candidates по индексу.
    ranked = sorted(
        zip(candidates, (float(s) for s in scores)),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top, top_score = ranked[0]

    info = {
        "elapsed_s": round(time.time() - started, 2),
        "n_candidates": len(candidates),
        "reranker_model": reranker.model_name,
    }

    return StageOutcome(
        stage=StageName.ANCHOR,
        node_ids=[top["node_id"]],
        payload={
            "anchor": top["node_id"],
            # Для совместимости с LLM-веткой держим то же поле reason —
            # debug-отчёт его читает напрямую.
            "reason": f"reranker top-1 (score={top_score:.4f})",
            "fallback": False,
            "selector": "reranker",
            "top_score": round(top_score, 6),
            # Полный ранжированный список для дебага. Намеренно компактно —
            # один список из (node_id, score), без лишних обёрток.
            "scores": [
                [c["node_id"], round(s, 6)] for c, s in ranked
            ],
        },
        info=info,
    )
