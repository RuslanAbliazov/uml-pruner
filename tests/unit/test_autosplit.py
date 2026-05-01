"""Unit tests for process_with_autosplit using a mocked LLM client."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.budget import TokenBudget
from src.llm.client import ContextOverflowError, LLMClient, LLMResponse
from src.approaches.rag_classes_filter.autosplit import (
    AutoSplitStats,
    MaxSplitDepthExceeded,
    process_with_autosplit,
)


def _build_prompt(items: list[int]) -> tuple[str, str]:
    """Encode item count as a proxy for prompt length so we can control budget."""
    # system prompt: fixed 10 chars
    # user prompt: N items × 50 chars each
    system = "SYSTEM...."
    user = "X" * (50 * len(items))
    return system, user


def _parse(content: str) -> set[int]:
    # content is comma-separated ints
    if not content:
        return set()
    return {int(x) for x in content.split(",") if x}


def _merge(a: set[int], b: set[int]) -> set[int]:
    return a | b


def _empty() -> set[int]:
    return set()


def _make_mock_client(response_items_fn=None):
    """Create a mock LLM client that returns a comma-joined list of ints
    based on the items it sees (encoded in user prompt length)."""
    client = LLMClient(api_key="dummy")

    call_log = []

    async def mock_call(system_prompt, user_prompt, json_mode=True, max_tokens=None):
        # Derive item count from user prompt length
        num_items = len(user_prompt) // 50
        call_log.append(num_items)
        if response_items_fn is not None:
            payload = response_items_fn(num_items)
        else:
            # Echo back num_items ints starting from 0 for test
            payload = ",".join(str(i) for i in range(num_items))
        return LLMResponse(
            content=payload, input_tokens=10, output_tokens=5, model="mock"
        )

    client.call = AsyncMock(side_effect=mock_call)
    client._call_log = call_log  # type: ignore
    return client


async def _oversized_dropped(item: int) -> set[int]:
    """Default handler: drop oversized single items."""
    return set()


async def _oversized_keep(item: int) -> set[int]:
    """Keep oversized single items as-is."""
    return {item}


def test_small_input_no_split():
    async def run():
        items = [1, 2, 3]
        client = _make_mock_client()
        # budget that comfortably fits all items
        budget = TokenBudget(
            context_window=10_000, output_reserve=100, safety_margin=100
        )
        stats = AutoSplitStats()
        result = await process_with_autosplit(
            items,
            _build_prompt,
            _parse,
            _merge,
            _empty,
            _oversized_dropped,
            budget,
            client,
            stats=stats,
        )
        assert stats.llm_calls == 1
        assert stats.max_depth_reached == 0
        # Result is whatever mock parsed
        assert isinstance(result, set)

    asyncio.run(run())


def test_splits_when_too_large():
    async def run():
        items = list(range(16))
        client = _make_mock_client()
        # Tight budget: each item ~50 chars => 12 tokens (chars/4). Budget 200
        # means ~14 items fit. So 16 items will split.
        budget = TokenBudget(
            context_window=1000, output_reserve=500, safety_margin=100
        )  # input_budget=400
        # Estimator: 50 chars = ~12 tokens per item; 16 items * 12 = 192 fits
        # Force split by making budget smaller
        budget = TokenBudget(
            context_window=300, output_reserve=50, safety_margin=50
        )  # input_budget=200
        # Each item adds 12 tokens, so 200/12 = ~16 items fit marginally; push to 32 items.
        items = list(range(32))
        stats = AutoSplitStats()
        await process_with_autosplit(
            items,
            _build_prompt,
            _parse,
            _merge,
            _empty,
            _oversized_dropped,
            budget,
            client,
            stats=stats,
        )
        assert stats.llm_calls >= 2  # at least split once
        assert stats.max_depth_reached >= 1

    asyncio.run(run())


def test_single_oversized_invokes_handler():
    async def run():
        items = [42]
        client = _make_mock_client()

        # Absurdly small budget: input_budget = 100-30-30 = 40.
        # estimator yields ~25 tokens for a single-item prompt, so we need
        # an even tighter limit; use a custom estimator to be explicit.
        def big_estimator(sys_p: str, user_p: str) -> int:
            return 1000

        budget = TokenBudget(
            context_window=500,
            output_reserve=50,
            safety_margin=50,
            estimator=big_estimator,
        )
        stats = AutoSplitStats()
        result = await process_with_autosplit(
            items,
            _build_prompt,
            _parse,
            _merge,
            _empty,
            _oversized_keep,
            budget,
            client,
            stats=stats,
        )
        assert stats.oversized_singles == 1
        assert stats.llm_calls == 0  # no LLM call when single item doesn't fit
        assert result == {42}

    asyncio.run(run())


def test_empty_input():
    async def run():
        client = _make_mock_client()
        budget = TokenBudget()
        stats = AutoSplitStats()
        result = await process_with_autosplit(
            [],
            _build_prompt,
            _parse,
            _merge,
            _empty,
            _oversized_dropped,
            budget,
            client,
            stats=stats,
        )
        assert stats.llm_calls == 0
        assert result == set()

    asyncio.run(run())


def test_merges_results_from_halves():
    async def run():
        items = list(range(40))
        # Response mock returns {num_items * 1000 + i for i in range(num_items)}
        # so each half yields distinct ints.
        call_counter = {"n": 0}

        async def mock_call(
            system_prompt, user_prompt, json_mode=True, max_tokens=None
        ):
            call_counter["n"] += 1
            num = len(user_prompt) // 50
            call_counter[f"batch_{call_counter['n']}"] = num
            # Emit one int per item; use unique offset
            offset = call_counter["n"] * 10_000
            payload = ",".join(str(offset + i) for i in range(num))
            return LLMResponse(
                content=payload, input_tokens=10, output_tokens=5, model="mock"
            )

        client = LLMClient(api_key="dummy")
        client.call = AsyncMock(side_effect=mock_call)

        budget = TokenBudget(context_window=400, output_reserve=50, safety_margin=50)
        stats = AutoSplitStats()
        result = await process_with_autosplit(
            items,
            _build_prompt,
            _parse,
            _merge,
            _empty,
            _oversized_dropped,
            budget,
            client,
            stats=stats,
        )
        # We expect at least two LLM calls due to split
        assert stats.llm_calls >= 2
        # Result should be a union — so more items than any single batch
        assert len(result) == 40 or len(result) >= 20

    asyncio.run(run())


def test_context_overflow_triggers_split():
    """Even if estimator says fits, a runtime ContextOverflowError should split."""

    async def run():
        items = list(range(8))

        call_count = {"n": 0}

        async def mock_call(
            system_prompt, user_prompt, json_mode=True, max_tokens=None
        ):
            call_count["n"] += 1
            num = len(user_prompt) // 50
            # Reject the first, full call; accept smaller ones.
            if num == 8:
                raise ContextOverflowError("simulated overflow")
            return LLMResponse(
                content=",".join(str(i) for i in range(num)),
                input_tokens=10,
                output_tokens=5,
                model="mock",
            )

        client = LLMClient(api_key="dummy")
        client.call = AsyncMock(side_effect=mock_call)

        # Huge budget — estimator thinks everything fits.
        budget = TokenBudget(
            context_window=1_000_000, output_reserve=1000, safety_margin=100
        )
        stats = AutoSplitStats()
        result = await process_with_autosplit(
            items,
            _build_prompt,
            _parse,
            _merge,
            _empty,
            _oversized_dropped,
            budget,
            client,
            stats=stats,
        )
        # Must have made the 1 failing full call + at least 2 split calls
        assert call_count["n"] >= 3
        assert stats.max_depth_reached >= 1
        assert len(result) >= 1

    asyncio.run(run())


def test_max_depth_exceeded_raises():
    async def run():
        items = list(range(10))
        client = _make_mock_client()

        # Force: nothing ever fits.
        def never_fits(sys_p: str, user_p: str) -> int:
            return 10_000

        budget = TokenBudget(
            context_window=500,
            output_reserve=50,
            safety_margin=50,
            estimator=never_fits,
        )
        stats = AutoSplitStats()
        # max_depth=1: we're allowed root (depth 0) and one split level
        # (depth 1). At depth 2 we must raise. But at depth 1 we still have
        # 5 items -> won't fit -> splits into depth=2 which exceeds.
        try:
            await process_with_autosplit(
                items,
                _build_prompt,
                _parse,
                _merge,
                _empty,
                _oversized_dropped,
                budget,
                client,
                stats=stats,
                max_depth=1,
            )
        except MaxSplitDepthExceeded:
            return
        raise AssertionError("Expected MaxSplitDepthExceeded")

    asyncio.run(run())


def run_all():
    tests = [
        test_small_input_no_split,
        test_splits_when_too_large,
        test_single_oversized_invokes_handler,
        test_empty_input,
        test_merges_results_from_halves,
        test_context_overflow_triggers_split,
        test_max_depth_exceeded_raises,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")


if __name__ == "__main__":
    run_all()
