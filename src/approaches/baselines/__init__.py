"""Query-agnostic baseline approaches.

These exist to give every "real" approach a sanity floor / ceiling. They are
deliberately simple, deterministic (where applicable), and require neither
LLM nor embedding-model dependencies — so they can run in CI without any
external services.

Registered names (in ``src.approaches.__init__.REGISTRY``):

    * ``empty``         — predict nothing. Recall = 0, precision undefined → 0.
    * ``full_diagram``  — predict the entire diagram. Recall = 1.0; precision
                          tiny. Useful as the F1 floor for "no filtering at all".
    * ``random_subset`` — predict K nodes chosen uniformly at random.
                          Seeded per ``sample_id`` for reproducibility.
    * ``top_degree``    — predict the top-K nodes by total degree (in + out).
                          Query-agnostic graph centrality baseline.

Configuration (all optional, in ``configs/config.yaml`` under
``approaches.<name>``):

    approaches:
      random_subset:
        size: 5            # default ≈ median(|gold|) on the current dataset
        seed: 42           # base seed; per-sample seed = (seed, sample_id)
      top_degree:
        size: 5            # how many top-degree nodes to keep

Oracle baselines (``central_plus_neighbors``, ``gold_only``) live in
``src/eval/oracle_baselines.py`` instead of here, because they intentionally
violate the "no ground-truth in the runner" invariant and therefore do NOT
implement the ``ApproachRunner`` protocol.
"""

from __future__ import annotations

from typing import Any

from src.approaches.baselines.runner import (
    EmptyBaseline,
    FullDiagramBaseline,
    RandomSubsetBaseline,
    TopDegreeBaseline,
)


def build_empty(cfg: Any | None = None) -> EmptyBaseline:
    return EmptyBaseline()


def build_full_diagram(cfg: Any | None = None) -> FullDiagramBaseline:
    return FullDiagramBaseline()


def _read_int(cfg: Any, section: str, key: str, default: int) -> int:
    """Pull ``approaches.<section>.<key>`` from cfg, falling back to ``default``.

    Tolerant: works with both ``Config``-wrapped and plain-dict configs, and
    accepts a missing ``approaches`` section entirely (returns the default).
    """
    if cfg is None:
        return default
    approaches = cfg.get("approaches") if hasattr(cfg, "get") else None
    if approaches is None:
        return default
    own = approaches.get(section) if hasattr(approaches, "get") else None
    if own is None:
        return default
    raw = own.get(key) if hasattr(own, "get") else None
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def build_random_subset(cfg: Any | None = None) -> RandomSubsetBaseline:
    size = _read_int(cfg, "random_subset", "size", default=5)
    seed = _read_int(cfg, "random_subset", "seed", default=42)
    return RandomSubsetBaseline(size=size, seed=seed)


def build_top_degree(cfg: Any | None = None) -> TopDegreeBaseline:
    size = _read_int(cfg, "top_degree", "size", default=5)
    return TopDegreeBaseline(size=size)


__all__ = [
    "EmptyBaseline",
    "FullDiagramBaseline",
    "RandomSubsetBaseline",
    "TopDegreeBaseline",
    "build_empty",
    "build_full_diagram",
    "build_random_subset",
    "build_top_degree",
]
