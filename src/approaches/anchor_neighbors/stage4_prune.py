"""Этап 4 — LLM классифицирует узлы подграфа как REQUIRED / USEFUL / IRRELEVANT.

Один LLM-вызов на подграф (anchor + соседи) → {required, useful}.

Контракт:
- `payload.required` / `payload.useful` — отфильтрованные по valid_ids списки.
- `node_ids` = required ∪ useful.
- Если вызов или парсинг упали — возвращается `StageOutcome(aborted=...)`.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

from src.approaches.anchor_neighbors import prompt_templates
from src.approaches.anchor_neighbors.llm_trace import LLMTracer
from src.approaches.anchor_neighbors.stage_outputs import StageName, StageOutcome
from src.llm.client import LLMClient
from src.llm.parser import parse_json_response


async def prune_subgraph(
    *,
    query: str,
    sub_nodes: list[dict[str, Any]],
    sub_edges: list[dict[str, Any]],
    llm: LLMClient,
    tracer: LLMTracer | None = None,
    sample_id: str = "",
) -> StageOutcome:
    """Один LLM-вызов: подграф → required / useful."""
    system_prompt = prompt_templates.prune_system()
    user_prompt = prompt_templates.prune_user(
        query=query,
        nodes=sub_nodes,
        edges=sub_edges,
    )

    if tracer is not None and sample_id:
        tracer.record_request(StageName.PRUNE, sample_id, system_prompt, user_prompt)

    started = time.time()
    try:
        resp = await llm.call(system_prompt, user_prompt, json_mode=True)
    except Exception as e:  # noqa: BLE001
        tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        if tracer is not None and sample_id:
            tracer.record_error(StageName.PRUNE, sample_id, tb_str)
        return StageOutcome(
            stage=StageName.PRUNE,
            aborted="llm_call_failed",
            info={
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": tb_str,
                "elapsed_s": round(time.time() - started, 2),
            },
        )

    if tracer is not None and sample_id:
        tracer.record_response(StageName.PRUNE, sample_id, resp.content)

    info: dict[str, Any] = {
        "elapsed_s": round(time.time() - started, 2),
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "subgraph_input_size": len(sub_nodes),
    }

    try:
        data = parse_json_response(resp.content)
    except ValueError as e:
        return StageOutcome(
            stage=StageName.PRUNE,
            aborted="bad_json",
            info={**info, "error": str(e),
                  "raw_excerpt": resp.content[:200],
                  "full_response": resp.content},
        )
    if not isinstance(data, dict):
        return StageOutcome(
            stage=StageName.PRUNE,
            aborted="bad_json_shape",
            info={**info, "raw_excerpt": resp.content[:200]},
        )

    valid_ids = {n["node_id"] for n in sub_nodes if n.get("node_id")}
    required = {
        x for x in (data.get("required") or [])
        if isinstance(x, str) and x in valid_ids
    }
    useful = {
        x for x in (data.get("useful") or [])
        if isinstance(x, str) and x in valid_ids and x not in required
    }

    reasoning = data.get("reasoning", "")
    if reasoning:
        info["reasoning"] = reasoning

    keep = required | useful
    return StageOutcome(
        stage=StageName.PRUNE,
        node_ids=sorted(keep),
        payload={"required": sorted(required), "useful": sorted(useful)},
        info=info,
    )
