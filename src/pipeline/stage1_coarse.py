"""Stage 1: Package-level coarse filtering.

Groups classes by package and asks the LLM which packages are likely relevant
to the user query. Returns the set of node_ids surviving this coarse pass.

Uses the autosplit driver to handle oversized batches: if a batch doesn't fit
the token budget, it is split recursively.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.llm.budget import TokenBudget
from src.llm.client import LLMClient
from src.llm.parser import parse_json_response
from src.llm.prompts import build_stage1_user_prompt, stage1_system_prompt
from src.pipeline.autosplit import (
    AutoSplitStats,
    MaxSplitDepthExceeded,
    process_with_autosplit,
)
from src.preprocessing.batching import make_batches
from src.preprocessing.package_grouper import group_by_package, summarize_packages
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _parse_stage1_response(content: str) -> set[str]:
    """Extract HIGH ∪ MEDIUM package names from a Stage 1 response."""
    try:
        parsed = parse_json_response(content)
    except ValueError:
        logger.warning("Stage 1: could not parse response; treating as empty")
        return set()
    if not isinstance(parsed, dict):
        return set()
    names: set[str] = set()
    for name in parsed.get("high") or []:
        if isinstance(name, str):
            names.add(name)
    for name in parsed.get("medium") or []:
        if isinstance(name, str):
            names.add(name)
    return names


async def _handle_oversized_package(pkg: dict[str, Any]) -> set[str]:
    """Fallback when a single-package prompt still doesn't fit.

    Extremely unlikely in practice. We optimistically keep the package name
    as relevant rather than silently dropping the signal.
    """
    logger.warning(
        "Stage 1: single package '%s' did not fit token budget; keeping it as relevant",
        pkg.get("name", "?"),
    )
    return {pkg["name"]}


async def run_stage1(
    query: str,
    nodes: list[dict[str, Any]],
    llm_client: LLMClient,
    budget: TokenBudget,
    batch_size: int = 40,
    max_parallel: int = 5,
    max_split_depth: int = 8,
) -> set[str]:
    """Run Stage 1 and return the surviving node_ids.

    Args:
        query: User query.
        nodes: All nodes of the diagram.
        llm_client: LLM client.
        budget: Token budget for overflow protection.
        batch_size: Initial package batch size.
        max_parallel: Concurrency between top-level batches (not within a
            single batch's autosplit tree, which is sequential).
        max_split_depth: Hard recursion cap for autosplit.
    """
    logger.info("Stage 1: grouping %d nodes by package", len(nodes))
    groups = group_by_package(nodes)
    summaries = summarize_packages(groups)
    logger.info("Stage 1: %d packages, batching by %d", len(summaries), batch_size)

    if len(summaries) <= 5:
        logger.info("Stage 1: skipped (only %d packages)", len(summaries))
        return {n["node_id"] for n in nodes}

    batches = make_batches(summaries, batch_size)

    def build_prompt(items: list[dict[str, Any]]) -> tuple[str, str]:
        # batch_idx / total_batches are informational — we pass (1,1) because
        # within an autosplit tree we're always working on a sub-slice.
        return (
            stage1_system_prompt(),
            build_stage1_user_prompt(query, items, 1, 1),
        )

    semaphore = asyncio.Semaphore(max_parallel)

    async def _run_single_batch(batch: list[dict[str, Any]]) -> set[str]:
        async with semaphore:
            stats = AutoSplitStats()
            try:
                result = await process_with_autosplit(
                    items=batch,
                    build_prompt_fn=build_prompt,
                    parse_response_fn=_parse_stage1_response,
                    merge_fn=lambda a, b: a | b,
                    empty_result_fn=set,
                    handle_oversized_single=_handle_oversized_package,
                    budget=budget,
                    llm_client=llm_client,
                    max_depth=max_split_depth,
                    stats=stats,
                )
            except MaxSplitDepthExceeded:
                # Graceful degradation: keep all packages in this batch as
                # potentially relevant. Downstream stages will re-evaluate.
                logger.warning(
                    "Stage 1: batch hit max_split_depth; keeping all packages as relevant"
                )
                return {p["name"] for p in batch if isinstance(p, dict)}
            if stats.max_depth_reached > 0:
                logger.info(
                    "Stage 1: batch triggered split (depth=%d, calls=%d)",
                    stats.max_depth_reached,
                    stats.llm_calls,
                )
            return result

    logger.info("Stage 1: dispatching %d top-level batches", len(batches))
    per_batch_results = await asyncio.gather(*(_run_single_batch(b) for b in batches))

    relevant_packages: set[str] = set()
    for r in per_batch_results:
        relevant_packages |= r
    logger.info("Stage 1: %d relevant packages selected", len(relevant_packages))

    surviving: set[str] = set()
    for name, members in groups.items():
        if name in relevant_packages:
            for n in members:
                surviving.add(n["node_id"])

    logger.info(
        "Stage 1: %d / %d nodes survived (%.1f%%)",
        len(surviving),
        len(nodes),
        100.0 * len(surviving) / max(len(nodes), 1),
    )

    if not surviving:
        logger.warning("Stage 1 filtered out everything — falling back to all nodes")
        return {n["node_id"] for n in nodes}

    return surviving
