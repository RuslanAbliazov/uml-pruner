"""Stage 2: Class-level refinement.

Takes the Stage 1 survivors and asks the LLM to classify each class as
REQUIRED, USEFUL, or IRRELEVANT given the query. REQUIRED and USEFUL classes
are returned separately so that Stage 3 can fall back to REQUIRED-only if
the context overflows.

Uses the autosplit driver to recursively split oversized batches.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from src.llm.budget import TokenBudget
from src.llm.client import LLMClient
from src.llm.parser import parse_json_response
from src.llm.prompts import build_stage2_user_prompt, stage2_system_prompt
from src.pipeline.autosplit import (
    AutoSplitStats,
    MaxSplitDepthExceeded,
    process_with_autosplit,
)
from src.preprocessing.batching import make_batches
from src.preprocessing.compressor import build_class_representation, build_edge_index
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Stage2Result:
    """Classes kept by Stage 2, split by label."""

    required: set[str] = field(default_factory=set)
    useful: set[str] = field(default_factory=set)

    @property
    def all_kept(self) -> set[str]:
        return self.required | self.useful

    def merge(self, other: "Stage2Result") -> "Stage2Result":
        return Stage2Result(
            required=self.required | other.required,
            useful=self.useful | other.useful,
        )


def _parse_stage2_response(content: str) -> Stage2Result:
    """Parse a single batch response into a Stage2Result."""
    try:
        parsed = parse_json_response(content)
    except ValueError:
        logger.warning("Stage 2: could not parse response; treating as empty")
        return Stage2Result()
    if not isinstance(parsed, dict):
        return Stage2Result()
    required = {nid for nid in (parsed.get("required") or []) if isinstance(nid, str)}
    useful = {nid for nid in (parsed.get("useful") or []) if isinstance(nid, str)}
    return Stage2Result(required=required, useful=useful)


def _build_rep_at_level(
    node: dict[str, Any],
    outgoing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    level: str,
) -> dict[str, Any]:
    return build_class_representation(node, outgoing, incoming, compact_level=level)


async def _try_oversized_ultra(
    rep: dict[str, Any],
    original_node: dict[str, Any],
    outgoing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    build_prompt_fn,
    budget: TokenBudget,
) -> Stage2Result:
    """When a single class is too big, retry with ultra-compact encoding.

    If even ultra doesn't fit, log a warning and drop the class.
    """
    ultra = _build_rep_at_level(original_node, outgoing, incoming, "ultra")
    system, user = build_prompt_fn([ultra])
    if budget.fits(system, user):
        # Return a marker result that the caller will still need to evaluate.
        # We do NOT call the LLM here — the autosplit driver's handler should
        # be simple and deterministic. Instead, we optimistically classify
        # this ultra-rep as USEFUL (conservative: better to keep it than drop).
        logger.info(
            "Stage 2: oversized class %s compressed to ultra, keeping as USEFUL",
            rep.get("node_id", "?"),
        )
        return Stage2Result(useful={rep["node_id"]})
    logger.warning(
        "Stage 2: class %s does not fit even in ultra mode; dropping",
        rep.get("node_id", "?"),
    )
    return Stage2Result()


async def run_stage2(
    query: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    surviving_ids: set[str],
    llm_client: LLMClient,
    budget: TokenBudget,
    batch_size: int = 120,
    max_parallel: int = 3,
    max_output: int = 500,
    max_split_depth: int = 8,
) -> Stage2Result:
    """Run Stage 2 and return REQUIRED + USEFUL node_ids."""
    candidate_nodes = [n for n in nodes if n["node_id"] in surviving_ids]
    logger.info("Stage 2: %d candidate nodes", len(candidate_nodes))
    if not candidate_nodes:
        return Stage2Result()

    outgoing, incoming = build_edge_index(edges)

    # Build compact representations once; we also keep a lookup so we can
    # rebuild ultra versions if needed for the oversized handler.
    node_by_id = {n["node_id"]: n for n in candidate_nodes}
    representations = [
        _build_rep_at_level(
            n, outgoing.get(n["node_id"], []), incoming.get(n["node_id"], []), "compact"
        )
        for n in candidate_nodes
    ]

    batches = make_batches(representations, batch_size)

    def build_prompt(items: list[dict[str, Any]]) -> tuple[str, str]:
        return (
            stage2_system_prompt(),
            build_stage2_user_prompt(query, items, 1, 1),
        )

    async def handle_oversized_rep(rep: dict[str, Any]) -> Stage2Result:
        nid = rep["node_id"]
        original = node_by_id.get(nid)
        if original is None:
            return Stage2Result()
        return await _try_oversized_ultra(
            rep,
            original,
            outgoing.get(nid, []),
            incoming.get(nid, []),
            build_prompt,
            budget,
        )

    semaphore = asyncio.Semaphore(max_parallel)

    async def _run_single_batch(batch: list[dict[str, Any]]) -> Stage2Result:
        async with semaphore:
            stats = AutoSplitStats()
            try:
                result = await process_with_autosplit(
                    items=batch,
                    build_prompt_fn=build_prompt,
                    parse_response_fn=_parse_stage2_response,
                    merge_fn=lambda a, b: a.merge(b),
                    empty_result_fn=Stage2Result,
                    handle_oversized_single=handle_oversized_rep,
                    budget=budget,
                    llm_client=llm_client,
                    max_depth=max_split_depth,
                    stats=stats,
                )
            except MaxSplitDepthExceeded:
                # Graceful degradation: treat all classes in this batch as
                # USEFUL so Stage 3 still has something to work with.
                logger.warning(
                    "Stage 2: batch hit max_split_depth; keeping all as USEFUL"
                )
                return Stage2Result(
                    useful={r["node_id"] for r in batch if isinstance(r, dict)}
                )
            if stats.max_depth_reached > 0:
                logger.info(
                    "Stage 2: batch triggered split (depth=%d, calls=%d, oversized=%d)",
                    stats.max_depth_reached,
                    stats.llm_calls,
                    stats.oversized_singles,
                )
            return result

    logger.info("Stage 2: dispatching %d top-level batches", len(batches))
    per_batch = await asyncio.gather(*(_run_single_batch(b) for b in batches))

    combined = Stage2Result()
    for r in per_batch:
        combined = combined.merge(r)

    # Restrict to surviving ids only (defensive).
    surviving_strict = {n["node_id"] for n in candidate_nodes}
    combined = Stage2Result(
        required=combined.required & surviving_strict,
        useful=combined.useful & surviving_strict,
    )

    logger.info(
        "Stage 2: %d REQUIRED, %d USEFUL (%d total)",
        len(combined.required),
        len(combined.useful),
        len(combined.all_kept),
    )

    # Enforce max_output: keep all REQUIRED, trim USEFUL.
    total = combined.all_kept
    if len(total) > max_output:
        logger.info(
            "Stage 2: over budget (%d > %d), trimming useful set",
            len(total),
            max_output,
        )
        budget_for_useful = max(0, max_output - len(combined.required))
        if budget_for_useful == 0:
            # REQUIRED alone is already at/above max_output — keep all
            # REQUIRED, drop all USEFUL.
            combined = Stage2Result(required=combined.required, useful=set())
        else:
            trimmed_useful = set(list(combined.useful)[:budget_for_useful])
            combined = Stage2Result(required=combined.required, useful=trimmed_useful)

    # Safety net: if everything got dropped, keep candidates as useful so
    # Stage 3 still has material to work with.
    if not combined.all_kept:
        logger.warning("Stage 2: empty output — falling back to candidates as useful")
        return Stage2Result(required=set(), useful=surviving_strict)

    return combined
