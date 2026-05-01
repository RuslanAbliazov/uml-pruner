"""Stage 2 — LLM picks the single best anchor from the RAG candidates.

If the LLM returns garbage we fall back to the top-1 candidate so the
pipeline still produces a result. If the LLM call itself blows up we
return ``(None, "")`` and let the runner abort the sample cleanly.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from src.core.logger import get_logger
from src.llm.client import LLMClient
from src.llm.parser import parse_json_response

from src.approaches.anchor_neighbors import prompts

logger = get_logger(__name__)


def _method_preview(methods: list[str], limit: int = 8) -> list[str]:
    """A few unique method names (no parameters) per candidate.

    Helps the LLM tell similar-looking classes apart without inflating
    the prompt.
    """
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


def _enrich_candidates(
    candidates: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for c in candidates:
        node = node_by_id.get(c["node_id"]) or {}
        enriched.append(
            {
                "node_id": c["node_id"],
                "name": c.get("name"),
                "type": c.get("type"),
                "score": c.get("score"),
                "methods_preview": _method_preview(node.get("methods") or []),
            }
        )
    return enriched


async def select_anchor(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    llm: LLMClient,
) -> tuple[Optional[str], str]:
    """Return ``(anchor_id, reason)``.

    ``anchor_id`` is None only when the LLM call itself fails or its
    response cannot be parsed. A *valid* response with a hallucinated
    ``node_id`` triggers a fallback to the top-1 candidate.
    """
    enriched = _enrich_candidates(candidates, node_by_id)
    user_prompt = prompts.select_user(query=query, candidates=enriched)

    t0 = time.time()
    try:
        resp = await llm.call(prompts.select_system(), user_prompt, json_mode=True)
    except Exception:
        logger.exception("anchor selection LLM call failed")
        return None, ""
    logger.debug(
        "anchor selection: %.2fs, %d in / %d out tokens",
        time.time() - t0, resp.input_tokens, resp.output_tokens,
    )

    try:
        data = parse_json_response(resp.content)
    except ValueError:
        logger.warning("anchor selection: could not parse LLM JSON")
        return None, ""
    if not isinstance(data, dict):
        return None, ""

    anchor = data.get("anchor")
    reason = (data.get("reason") or "").strip()
    valid_ids = {c["node_id"] for c in candidates}
    if isinstance(anchor, str) and anchor in valid_ids:
        return anchor, reason

    logger.warning(
        "anchor selection: LLM returned invalid/missing anchor (%r); "
        "falling back to top-1 candidate", anchor,
    )
    return candidates[0]["node_id"], "fallback: LLM returned invalid anchor"
