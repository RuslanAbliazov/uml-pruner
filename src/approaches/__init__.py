"""Pluggable UML-pruning approaches.

Every approach implements the :class:`ApproachRunner` protocol defined in
:mod:`src.core.types`. New approaches register themselves in the
:data:`REGISTRY` dictionary so that they can be selected by name from the
benchmark / CLI scripts.

Layout
------
::

    src/approaches/
    ├── __init__.py            # this file: registry + dispatch
    ├── base.py                # ApproachRunner protocol + ApproachResult
    ├── rag_classes_filter/    # #1: RAG batch + LLM filter (baseline)
    ├── anchor_neighbors/      # #2: anchor + neighbors + prune
    ├── agentic_chunks/        # #3: chunk diagram into files, agent picks
    └── human_like_agent/      # #4: agent picks N anchors, expands by
                               #     betweenness / calls_in_code

Each approach's package exposes:
    - ``build_runner(cfg) -> ApproachRunner``
    - ``NAME``: short string identifier used in the registry.
"""

from __future__ import annotations

from typing import Any, Callable

from src.core.types import ApproachInputs, ApproachResult, ApproachRunner

# Lazy registry: name -> factory(cfg) -> ApproachRunner.
# Factories are imported lazily so that an approach with optional dependencies
# (e.g. agent frameworks) doesn't break the registry import for other users.
_FactoryFn = Callable[[Any], ApproachRunner]


def _factory_rag_classes_filter() -> _FactoryFn:
    from src.approaches.rag_classes_filter.runner import build_runner

    return build_runner


def _factory_anchor_neighbors() -> _FactoryFn:
    from src.approaches.anchor_neighbors.runner import build_runner

    return build_runner


def _factory_agentic_chunks() -> _FactoryFn:
    from src.approaches.agentic_chunks.runner import build_runner

    return build_runner


def _factory_human_like_agent() -> _FactoryFn:
    from src.approaches.human_like_agent.runner import build_runner

    return build_runner


REGISTRY: dict[str, Callable[[], _FactoryFn]] = {
    "rag_classes_filter": _factory_rag_classes_filter,
    "anchor_neighbors": _factory_anchor_neighbors,
    "agentic_chunks": _factory_agentic_chunks,
    "human_like_agent": _factory_human_like_agent,
}


def list_approaches() -> list[str]:
    """Return registered approach names."""
    return sorted(REGISTRY.keys())


def get_runner(name: str, cfg: Any | None = None) -> ApproachRunner:
    """Instantiate the runner for the given approach name."""
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown approach '{name}'. Available: {', '.join(list_approaches())}"
        )
    factory = REGISTRY[name]()
    return factory(cfg)


__all__ = [
    "ApproachInputs",
    "ApproachResult",
    "ApproachRunner",
    "REGISTRY",
    "list_approaches",
    "get_runner",
]
