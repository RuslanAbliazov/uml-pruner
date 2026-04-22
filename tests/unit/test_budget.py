"""Unit tests for TokenBudget."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.budget import TokenBudget


def test_input_budget_math():
    b = TokenBudget(context_window=128_000, output_reserve=4_096, safety_margin=2_000)
    assert b.input_budget == 128_000 - 4_096 - 2_000


def test_fits_small_prompt():
    b = TokenBudget()
    assert b.fits("You are an assistant.", "Hello")


def test_fits_with_custom_estimator():
    def const_estimator(sys_p: str, user_p: str) -> int:
        return 100

    b = TokenBudget(
        context_window=1000,
        output_reserve=200,
        safety_margin=100,
        estimator=const_estimator,
    )
    # input_budget = 700
    assert b.input_budget == 700
    assert b.fits("", "")  # 100 <= 700
    assert b.tokens_used("", "") == 100
    assert b.remaining("", "") == 600


def test_does_not_fit():
    def big_estimator(sys_p: str, user_p: str) -> int:
        return 10_000

    b = TokenBudget(
        context_window=5000,
        output_reserve=1000,
        safety_margin=500,
        estimator=big_estimator,
    )
    assert b.input_budget == 3500
    assert not b.fits("", "")
    assert b.remaining("", "") == 3500 - 10_000


def test_invalid_budget_raises():
    b = TokenBudget(context_window=100, output_reserve=200, safety_margin=0)
    try:
        _ = b.input_budget
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-positive budget")


def run_all():
    tests = [
        test_input_budget_math,
        test_fits_small_prompt,
        test_fits_with_custom_estimator,
        test_does_not_fit,
        test_invalid_budget_raises,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")


if __name__ == "__main__":
    run_all()
