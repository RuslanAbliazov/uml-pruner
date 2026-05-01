"""Approach #1: RAG batch + LLM classifier (baseline).

Implementation lives in :mod:`runner`. See ``README.md`` in
``scripts/approaches/rag_classes_filter/`` for the user-facing description.
"""

from src.approaches.rag_classes_filter.runner import NAME, build_runner

__all__ = ["NAME", "build_runner"]
