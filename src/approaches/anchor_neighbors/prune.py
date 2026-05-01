"""Stage 4 — LLM classifies the subgraph as REQUIRED / USEFUL / IRRELEVANT.

The user-facing payload is intentionally narrow: the LLM never sees the
free-form ``description`` field (it's LLM-generated upstream and could
leak the answer), and the methods/params lists are capped.
"""

from __future__ import annotations

import time
from typing import Any

from src.core.logger import get_logger
from src.llm.client import LLMClient
from src.llm.parser import parse_json_response

from src.approaches.anchor_neighbors import prompts
from src.approaches.anchor_neighbors.candidates import short_name

logger = get_logger(__name__)


def _node_for_llm(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": node.get("node_id"),
        "name": node.get("name") or short_name(node.get("node_id", "")),
        "type": node.get("type", "class"),
        "methods": (node.get("methods") or [])[:30],
        "params": (node.get("params") or [])[:20],
    }


def _edge_for_llm(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        "from": edge.get("node_id_from"),
        "to": edge.get("node_id_to"),
        "kind": edge.get("description") or edge.get("kind") or "",
    }


async def prune(
    *,
    query: str,
    anchor: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    llm: LLMClient,
) -> tuple[set[str], set[str]]:
    """Return ``(required_ids, useful_ids)`` restricted to ``nodes``."""
    user_prompt = prompts.prune_user(
        query=query,
        anchor=anchor,
        nodes=[_node_for_llm(n) for n in nodes],
        edges=[_edge_for_llm(e) for e in edges],
    )
    t0 = time.time()
    try:
        resp = await llm.call(prompts.prune_system(), user_prompt, json_mode=True)
    except Exception:
        logger.exception("anchor prune LLM call failed")
        return set(), set()
    logger.debug(
        "anchor prune: %.2fs, %d in / %d out tokens",
        time.time() - t0, resp.input_tokens, resp.output_tokens,
    )

    try:
        data = parse_json_response(resp.content)
    except ValueError:
        logger.warning("anchor prune: could not parse LLM JSON")
        return set(), set()
    if not isinstance(data, dict):
        return set(), set()

    valid_ids = {n["node_id"] for n in nodes if n.get("node_id")}
    required = {
        x for x in (data.get("required") or [])
        if isinstance(x, str) and x in valid_ids
    }
    useful = {
        x for x in (data.get("useful") or [])
        if isinstance(x, str) and x in valid_ids and x not in required
    }
    return required, useful
