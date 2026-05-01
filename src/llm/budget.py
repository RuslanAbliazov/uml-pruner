"""Token budget management for LLM prompts.

The `TokenBudget` knows the model's context window and reserves a portion for
the response plus a safety margin. It exposes a single `fits()` method used
across the pipeline to decide whether a given (system, user) prompt pair will
be accepted by the API.

Token counting defaults to the fast chars/4 heuristic; the safety margin
compensates for its inaccuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.core.tokens import estimate_prompt_tokens_fast


@dataclass
class TokenBudget:
    """Manages the token budget for LLM calls.

    Attributes:
        context_window: Model's total context size (e.g. 128_000 for GPT-4 Turbo).
        output_reserve: Tokens reserved for the model's response.
        safety_margin: Extra tokens reserved to absorb chars/4 estimator error.
        estimator: Callable(system, user) -> int that estimates prompt tokens.
    """

    context_window: int = 128_000
    output_reserve: int = 4_096
    safety_margin: int = 2_000
    estimator: Callable[[str, str], int] = estimate_prompt_tokens_fast

    @property
    def input_budget(self) -> int:
        """Maximum tokens that can be spent on the input prompt."""
        budget = self.context_window - self.output_reserve - self.safety_margin
        if budget <= 0:
            raise ValueError(
                f"Non-positive input budget: context={self.context_window}, "
                f"output_reserve={self.output_reserve}, safety_margin={self.safety_margin}"
            )
        return budget

    def tokens_used(self, system: str, user: str) -> int:
        """Estimate tokens required for the given prompt pair."""
        return self.estimator(system, user)

    def fits(self, system: str, user: str) -> bool:
        """Return True iff the prompt fits within the input budget."""
        return self.tokens_used(system, user) <= self.input_budget

    def remaining(self, system: str, user: str) -> int:
        """How many input tokens remain after the prompt (may be negative)."""
        return self.input_budget - self.tokens_used(system, user)
