"""Batching utilities for splitting large inputs across LLM calls."""

from __future__ import annotations

from typing import Iterable, TypeVar

T = TypeVar("T")


def make_batches(items: list[T], batch_size: int) -> list[list[T]]:
    """Split a list into fixed-size chunks (last chunk may be smaller)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def chunked(iterable: Iterable[T], batch_size: int) -> Iterable[list[T]]:
    """Yield lists of size batch_size from an iterable."""
    buf: list[T] = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if buf:
        yield buf
