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
from src.approaches.anchor_neighbors.llm_trace import LLMTracer
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
    edges: list[dict[str, str]]
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
                "edges": [e for e in edges if e["node_id_from"] == c["node_id"] or e["node_id_to"] == c["node_id"]]
            }
        )
    return enriched


def _fill_anchors(
    primary: str,
    candidates: list[dict[str, Any]],
    n_anchors: int,
) -> list[str]:
    """Дополнить ``primary`` следующими по RAG-score кандидатами до N штук.

    ``candidates`` уже упорядочен по убыванию RAG-score (этап 1 это
    гарантирует). Дубликат ``primary`` исключаем.
    """
    anchors: list[str] = [primary]
    if n_anchors <= 1:
        return anchors
    for c in candidates:
        nid = c.get("node_id")
        if not nid or nid == primary:
            continue
        anchors.append(nid)
        if len(anchors) >= n_anchors:
            break
    return anchors


async def select_anchor(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    edges: list[dict[str, str]],
    llm: LLMClient,
    n_anchors: int = 1,
    tracer: LLMTracer | None = None,
    sample_id: str = "",
) -> StageOutcome:
    """Выбрать anchor через LLM, дополнить до ``n_anchors`` по RAG-score.

    Аргументы — ТОЛЬКО (repo, query)-уровневые данные (запрос + диаграмма).
    Никакого ground-truth тут нет и быть не должно.

    Текущий промпт LLM просит «pick ONE». Поэтому для multi-anchor
    режима (``n_anchors > 1``) мы:
      1) спрашиваем LLM один anchor (как раньше);
      2) добавляем top следующих кандидатов по RAG-score, пропуская
         выбранный LLM, до общего числа ``n_anchors``.

    При ``n_anchors == 1`` поведение полностью совпадает с прежним.

    Если передан `tracer` и непустой `sample_id`, перед вызовом LLM
    записываем последний request, после вызова — last response (или текст
    ошибки, если запрос упал). Файлы перезаписываются — на диске лежит
    только последний прогон по этому `sample_id`.
    """
    if not candidates:
        return StageOutcome(
            stage=StageName.ANCHOR,
            aborted="no_candidates",
            info={"reason": "stage 1 returned no candidates"},
        )

    payload_for_llm = _enrich_for_llm(candidates, node_by_id, edges)
    system_prompt = prompt_templates.anchor_selection_system()
    user_prompt = prompt_templates.anchor_selection_user(
        query=query, candidates=payload_for_llm
    )

    # Сохранить запрос ДО вызова LLM: даже если запрос упадёт по таймауту,
    # на диске останется ровно тот payload, что был отправлен.
    if tracer is not None and sample_id:
        tracer.record_request(StageName.ANCHOR, sample_id, system_prompt, user_prompt)

    started = time.time()
    try:
        resp = await llm.call(system_prompt, user_prompt, json_mode=True)
    except Exception as e:  # noqa: BLE001 — единая точка обработки внешних сбоев
        if tracer is not None and sample_id:
            tracer.record_error(StageName.ANCHOR, sample_id, repr(e))
        return StageOutcome(
            stage=StageName.ANCHOR,
            aborted="llm_call_failed",
            info={"error": repr(e), "elapsed_s": round(time.time() - started, 2)},
        )

    # Сохранить сырой ответ. Делаем сразу после возврата, до парсинга:
    # если парсинг упадёт — у нас на диске останется именно то, что
    # вернула модель.
    if tracer is not None and sample_id:
        tracer.record_response(StageName.ANCHOR, sample_id, resp.content)

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
        anchors = _fill_anchors(chosen, candidates, n_anchors)
        return StageOutcome(
            stage=StageName.ANCHOR,
            node_ids=list(anchors),
            payload={
                "anchor": chosen,
                "anchors": list(anchors),
                "reason": reason,
                "fallback": False,
            },
            info={**info, "n_anchors_returned": len(anchors)},
        )

    # Плохой случай: галлюцинация / пустой / не из списка. Берём top-1
    # RAG-кандидата как самый разумный fallback (порядок гарантирован
    # этапом 1 — кандидаты идут в порядке убывания score).
    fallback_id = candidates[0]["node_id"]
    anchors = _fill_anchors(fallback_id, candidates, n_anchors)
    return StageOutcome(
        stage=StageName.ANCHOR,
        node_ids=list(anchors),
        payload={
            "anchor": fallback_id,
            "anchors": list(anchors),
            "reason": reason or "<empty>",
            "fallback": True,
            "llm_returned": chosen,
        },
        info={**info, "n_anchors_returned": len(anchors)},
    )
