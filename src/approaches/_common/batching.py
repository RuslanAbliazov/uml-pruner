"""Tiny batching helper shared by approaches that send paged LLM calls."""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def make_batches(items: list[T], batch_size: int) -> list[list[T]]:
    """Split a list into fixed-size chunks (last chunk may be smaller)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
