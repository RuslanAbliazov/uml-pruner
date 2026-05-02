"""Этап 2 — LLM выбирает один anchor из RAG-кандидатов.

Возвращает StageOutcome с ровно одним `node_id`. Если LLM вернул что-то
непарсимое или несуществующее — мы **не** молча подставляем top-1, а
помечаем это в `payload.fallback`, чтобы при дебаге было видно, на каком
запросе модель ошиблась с выбором (это очень важно для оценки самого
этапа: top-1 RAG ≠ выбор LLM, и метрика должна различать эти случаи).

Если LLM-вызов вообще упал (сеть, таймаут, парсинг) — возвращаем
`aborted=llm_call_failed`. Pipeline останавливается; этап считается
неуспешным и метрики выставляются в нули.
"""

from __future__ import annotations

import time
from typing import Any

from src.approaches.anchor_neighbors import prompt_templates
from src.approaches.anchor_neighbors.stage_outputs import StageName, StageOutcome
from src.llm.client import LLMClient
from src.llm.parser import parse_json_response


def _method_preview(methods: list[str], limit: int = 8) -> list[str]:
    """Дать LLM по 8 коротких имён методов на класс — отличает похожие
    классы друг от друга, не раздувая промпт сигнатурами параметров."""
    seen: set[str] = set()
    out: list[str] = []
    for m in methods:
        head = m.split("(", 1)[0].strip()
        if not head or head in seen:
            continue
        seen.add(head)
        out.append(head)
        if len(out) >= limit:
            break
    return out


def _enrich_for_llm(
    candidates: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Сделать LLM-payload: id, name, type, score + сжатый список методов.

    Намеренно НЕ передаём поле `description`: оно сгенерировано LLM на
    более раннем этапе подготовки данных и может содержать формулировки,
    очень близкие к ground-truth — это была бы утечка."""
    enriched: list[dict[str, Any]] = []
    for c in candidates:
        node = node_by_id.get(c["node_id"]) or {}
        enriched.append(
            {
                "node_id": c["node_id"],
                "name": c.get("name"),
                "type": c.get("type"),
                "score": c.get("score"),
                "methods": node.get("methods"),
                "params": node.get("params"),
            }
        )
    return enriched


async def select_anchor(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    llm: LLMClient,
) -> StageOutcome:
    """Выбрать anchor через LLM. Возвращает StageOutcome.

    Аргументы — ТОЛЬКО (repo, query)-уровневые данные (запрос + диаграмма).
    Никакого ground-truth тут нет и быть не должно.
    """
    if not candidates:
        return StageOutcome(
            stage=StageName.ANCHOR,
            aborted="no_candidates",
            info={"reason": "stage 1 returned no candidates"},
        )

    payload_for_llm = _enrich_for_llm(candidates, node_by_id)
    user_prompt = prompt_templates.anchor_selection_user(
        query=query, candidates=payload_for_llm
    )

    started = time.time()
    try:
        resp = await llm.call(
            prompt_templates.anchor_selection_system(),
            user_prompt,
            json_mode=True,
        )
    except Exception as e:  # noqa: BLE001 — единая точка обработки внешних сбоев
        return StageOutcome(
            stage=StageName.ANCHOR,
            aborted="llm_call_failed",
            info={"error": repr(e), "elapsed_s": round(time.time() - started, 2)},
        )

    info = {
        "elapsed_s": round(time.time() - started, 2),
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
    }

    # Парсим ответ. JSON-режим обычно чистый, но мы всё равно толерантны.
    try:
        data = parse_json_response(resp.content)
    except ValueError:
        return StageOutcome(
            stage=StageName.ANCHOR,
            aborted="bad_json",
            info={**info, "raw_excerpt": resp.content[:200]},
        )
    if not isinstance(data, dict):
        return StageOutcome(
            stage=StageName.ANCHOR,
            aborted="bad_json_shape",
            info={**info, "raw_excerpt": resp.content[:200]},
        )

    valid_ids = {c["node_id"] for c in candidates}
    chosen = data.get("anchor")
    reason = (data.get("reason") or "").strip()

    # Хороший случай: LLM выбрал валидный из списка кандидатов.
    if isinstance(chosen, str) and chosen in valid_ids:
        return StageOutcome(
            stage=StageName.ANCHOR,
            node_ids=[chosen],
            payload={"anchor": chosen, "reason": reason, "fallback": False},
            info=info,
        )

    # Плохой случай: галлюцинация / пустой / не из списка. Берём top-1
    # RAG-кандидата как самый разумный fallback (порядок гарантирован
    # этапом 1 — кандидаты идут в порядке убывания score).
    fallback_id = candidates[0]["node_id"]
    return StageOutcome(
        stage=StageName.ANCHOR,
        node_ids=[fallback_id],
        payload={
            "anchor": fallback_id,
            "reason": reason or "<empty>",
            "fallback": True,
            "llm_returned": chosen,
        },
        info=info,
    )
