"""Generic recursive batch auto-split driver used by all stages.

Given a list of items and a function that renders them into a (system, user)
prompt pair, the driver:

1. Tries to send the whole batch if it fits into the TokenBudget.
2. If overflow is detected (either pre-flight via budget.fits() or at the API
   level via ContextOverflowError), splits the batch in half and recurses on
   each half sequentially.
3. When a single item still does not fit, delegates to the caller-provided
   `handle_oversized_single` hook to compress / drop it.
4. A hard recursion depth limit prevents infinite loops.

The driver is stage-agnostic: callers supply:
- `build_prompt_fn(items) -> (system, user)`
- `parse_response_fn(raw_content) -> R`   — turn raw LLM text into a stage
  result (e.g. {required, useful} sets).
- `merge_fn(left, right) -> R`            — combine two partial results.
- `empty_result_fn() -> R`                — identity element for merge, used
  when a branch collapses (empty batch / single item dropped).
- `handle_oversized_single(item) -> R`    — stage-specific fallback for a
  single item that cannot fit even alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, TypeVar

from src.llm.budget import TokenBudget
from src.llm.client import ContextOverflowError, LLMClient
from src.core.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")  # input item type
R = TypeVar("R")  # result type


@dataclass
class AutoSplitStats:
    """Telemetry about what happened during an autosplit run."""

    llm_calls: int = 0
    max_depth_reached: int = 0
    oversized_singles: int = 0
    skipped_single_items: int = 0

    def observe_depth(self, depth: int) -> None:
        if depth > self.max_depth_reached:
            self.max_depth_reached = depth


class MaxSplitDepthExceeded(Exception):
    """Raised when recursion hits the depth limit without fitting."""


async def process_with_autosplit(
    items: list[T],
    build_prompt_fn: Callable[[list[T]], tuple[str, str]],
    parse_response_fn: Callable[[str], R],
    merge_fn: Callable[[R, R], R],
    empty_result_fn: Callable[[], R],
    handle_oversized_single: Callable[[T], Awaitable[R]],
    budget: TokenBudget,
    llm_client: LLMClient,
    json_mode: bool = True,
    max_depth: int = 8,
    stats: AutoSplitStats | None = None,
    _depth: int = 0,
) -> R:
    """Recursively split `items` until prompts fit the budget, then merge.

    Args:
        items: Items to process.
        build_prompt_fn: (items) -> (system, user).
        parse_response_fn: (raw_content) -> R.
        merge_fn: combine two partial results.
        empty_result_fn: neutral result (for empty branches).
        handle_oversized_single: called when len(items) == 1 and the single
            item still doesn't fit. The callback decides how to degrade
            (compress, drop, etc.).
        budget: TokenBudget used for pre-flight checks.
        llm_client: LLM client used for actual calls.
        json_mode: Passed to llm_client.call.
        max_depth: Hard limit on recursion depth.
        stats: Optional stats collector.
        _depth: Internal recursion parameter.
    """
    if stats is None:
        stats = AutoSplitStats()
    stats.observe_depth(_depth)

    if not items:
        return empty_result_fn()

    if _depth > max_depth:
        logger.error(
            "autosplit: max_depth=%d exceeded with %d items — cannot split further",
            max_depth,
            len(items),
        )
        raise MaxSplitDepthExceeded(
            f"Could not fit prompt within {max_depth} split levels"
        )

    system_prompt, user_prompt = build_prompt_fn(items)

    # Pre-flight budget check first (cheap).
    fits = budget.fits(system_prompt, user_prompt)

    if fits:
        try:
            response = await llm_client.call(
                system_prompt, user_prompt, json_mode=json_mode
            )
            stats.llm_calls += 1
            return parse_response_fn(response.content)
        except ContextOverflowError:
            # Estimator was too optimistic — fall through to split logic below.
            logger.warning(
                "autosplit: fast estimator said fits=True but API rejected "
                "(depth=%d, items=%d). Splitting.",
                _depth,
                len(items),
            )
            fits = False

    # Prompt is too large. If we have a single item, invoke the fallback.
    if len(items) == 1:
        stats.oversized_singles += 1
        logger.warning(
            "autosplit: single item at depth=%d still too large; delegating "
            "to handle_oversized_single",
            _depth,
        )
        return await handle_oversized_single(items[0])

    # Split in half sequentially: left first, then right.
    mid = len(items) // 2
    logger.info(
        "autosplit: depth=%d splitting %d items -> %d + %d",
        _depth,
        len(items),
        mid,
        len(items) - mid,
    )

    left_result = await process_with_autosplit(
        items[:mid],
        build_prompt_fn,
        parse_response_fn,
        merge_fn,
        empty_result_fn,
        handle_oversized_single,
        budget,
        llm_client,
        json_mode=json_mode,
        max_depth=max_depth,
        stats=stats,
        _depth=_depth + 1,
    )
    right_result = await process_with_autosplit(
        items[mid:],
        build_prompt_fn,
        parse_response_fn,
        merge_fn,
        empty_result_fn,
        handle_oversized_single,
        budget,
        llm_client,
        json_mode=json_mode,
        max_depth=max_depth,
        stats=stats,
        _depth=_depth + 1,
    )
    return merge_fn(left_result, right_result)
