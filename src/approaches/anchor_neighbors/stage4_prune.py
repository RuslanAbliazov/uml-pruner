"""Этап 4 — LLM раскладывает подграф на REQUIRED / USEFUL / IRRELEVANT.

Что прокидываем модели:
* свободный пользовательский запрос;
* anchor — единственный класс, который изначально гарантированно релевантен;
* список всех узлов подграфа в усечённом виде (`_node_for_llm`);
* список рёбер подграфа в усечённом виде (`_edge_for_llm`).

Что НЕ прокидываем:
* поле `description` узла. Оно сгенерировано LLM на стадии подготовки
  данных и часто пересекается формулировками с эталоном — это утечка.
* полные сигнатуры (методов больше 30, параметров больше 20). Это и
  гигиена контекстного окна, и снижение поверхности утечки.

Контракт результата:
* `node_ids` — все классы, которые пайплайн в итоге оставляет в подграфе
  (т.е. `required ∪ useful`). Anchor дополнительно гарантируется в
  required, если LLM почему-то его не упомянул.
* `payload.required` / `payload.useful` — раздельные множества, нужны
  и для `to_diagram()`, и для метрик.
"""

from __future__ import annotations

import time
from typing import Any

from src.approaches.anchor_neighbors import prompt_templates
from src.approaches.anchor_neighbors.llm_trace import LLMTracer
from src.approaches.anchor_neighbors.stage_outputs import StageName, StageOutcome
from src.llm.client import LLMClient
from src.llm.parser import parse_json_response


_METHODS_PER_NODE = 30
_PARAMS_PER_NODE = 20


def _short_name(node_id: str) -> str:
    return node_id.rsplit(".", 1)[-1] if "." in node_id else node_id


def _node_for_llm(node: dict[str, Any]) -> dict[str, Any]:
    """Узкий снимок узла для LLM — без `description`, с обрезанными
    методами/параметрами."""
    return {
        "node_id": node.get("node_id"),
        "name": node.get("name") or _short_name(node.get("node_id", "")),
        "type": node.get("type", "class"),
        "methods": (node.get("methods") or []),
        "params": (node.get("params") or []),
    }


def _edge_for_llm(edge: dict[str, Any]) -> dict[str, Any]:
    """Узкий снимок ребра — оставляем только направление и тип связи."""
    return {
        "from": edge.get("node_id_from"),
        "to": edge.get("node_id_to"),
        "kind": edge.get("description") or edge.get("kind") or "",
    }


async def prune_subgraph(
    *,
    query: str,
    anchor: str,
    sub_nodes: list[dict[str, Any]],
    sub_edges: list[dict[str, Any]],
    llm: LLMClient,
    tracer: LLMTracer | None = None,
    sample_id: str = "",
) -> StageOutcome:
    """Прогнать LLM-прунинг и вернуть структурированный StageOutcome.

    Если передан `tracer` и `sample_id` — пишем последний request/response
    в `<root>/prune/<sample_id>.{req,resp}.txt` (перезаписывая прошлый).
    """
    system_prompt = prompt_templates.prune_system()
    user_prompt = prompt_templates.prune_user(
        query=query,
        anchor=anchor,
        nodes=[_node_for_llm(n) for n in sub_nodes],
        edges=[_edge_for_llm(e) for e in sub_edges],
    )

    if tracer is not None and sample_id:
        tracer.record_request(StageName.PRUNE, sample_id, system_prompt, user_prompt)

    started = time.time()
    try:
        resp = await llm.call(system_prompt, user_prompt, json_mode=True)
    except Exception as e:  # noqa: BLE001 — общая точка обработки внешних сбоев
        if tracer is not None and sample_id:
            tracer.record_error(StageName.PRUNE, sample_id, repr(e))
        return StageOutcome(
            stage=StageName.PRUNE,
            aborted="llm_call_failed",
            info={"error": repr(e), "elapsed_s": round(time.time() - started, 2)},
        )

    if tracer is not None and sample_id:
        tracer.record_response(StageName.PRUNE, sample_id, resp.content)

    info = {
        "elapsed_s": round(time.time() - started, 2),
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "subgraph_input_size": len(sub_nodes),
    }

    try:
        data = parse_json_response(resp.content)
    except ValueError:
        return StageOutcome(
            stage=StageName.PRUNE,
            aborted="bad_json",
            info={**info, "raw_excerpt": resp.content[:200]},
        )
    if not isinstance(data, dict):
        return StageOutcome(
            stage=StageName.PRUNE,
            aborted="bad_json_shape",
            info={**info, "raw_excerpt": resp.content[:200]},
        )

    # Доверяем только тем node_id, которые реально были в подграфе.
    valid_ids = {n["node_id"] for n in sub_nodes if n.get("node_id")}
    required = {
        x for x in (data.get("required") or [])
        if isinstance(x, str) and x in valid_ids
    }
    useful = {
        x for x in (data.get("useful") or [])
        if isinstance(x, str) and x in valid_ids and x not in required
    }

    # Anchor по определению релевантен — гарантируем его наличие.
    anchor_added = False
    if anchor not in required and anchor not in useful:
        required.add(anchor)
        anchor_added = True

    keep = required | useful
    return StageOutcome(
        stage=StageName.PRUNE,
        node_ids=sorted(keep),
        payload={
            "required": sorted(required),
            "useful": sorted(useful),
            "anchor_force_added_to_required": anchor_added,
        },
        info=info,
    )
