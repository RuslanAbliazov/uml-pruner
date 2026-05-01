"""Token counting utilities.

Two flavors are provided:
- Accurate (tiktoken) — correct but has ~1-10ms/KB overhead.
- Fast (chars/4 heuristic) — used during hot-path budget checks where we split
  batches. A safety margin in TokenBudget compensates for the underestimation.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=4)
def _get_encoder(model: str) -> tiktoken.Encoding:
    """Return a tiktoken encoding for a given model with fallback."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for newer / unknown models
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4-turbo-preview") -> int:
    """Count tokens in a string."""
    if not text:
        return 0
    encoder = _get_encoder(model)
    return len(encoder.encode(text))


def count_tokens_for_messages(
    messages: list[dict[str, str]],
    model: str = "gpt-4-turbo-preview",
) -> int:
    """Approximate token count for a chat-messages list.

    Uses a fixed per-message overhead of 4 tokens (OpenAI cookbook heuristic).
    """
    total = 0
    for msg in messages:
        total += 4
        for value in msg.values():
            total += count_tokens(str(value), model)
    total += 2  # reply priming
    return total


# -----------------------------------------------------------------------------
# Fast estimators (chars/4 heuristic)
# -----------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4


def estimate_tokens_fast(text: str) -> int:
    """Fast chars/4 token estimate.

    Underestimates for languages with long tokens (e.g. long identifiers), but
    this is acceptable for budget checks because TokenBudget reserves a safety
    margin. ~1000x faster than tiktoken.
    """
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_prompt_tokens_fast(system: str, user: str) -> int:
    """Fast estimate for a (system, user) chat prompt pair.

    Adds a constant per-message overhead similar to count_tokens_for_messages.
    """
    return (
        estimate_tokens_fast(system)
        + estimate_tokens_fast(user)
        + 4 * 2  # per-message overhead
        + 2  # reply priming
    )
