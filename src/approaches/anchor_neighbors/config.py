"""Approach config + factory.

Reads the YAML config and produces a configured ``AnchorNeighborsRunner``.
Config sections used:

* ``approaches.anchor_neighbors`` — approach-specific knobs.
* ``embeddings``                  — shared RAG defaults (model, device, ...).
* ``llm``                         — LLM connection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.llm.client import LLMClient


@dataclass
class AnchorNeighborsConfig:
    # ---- candidate generation (RAG) --------------------------------------
    n_candidates: int = 10
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "auto"
    embedding_batch_size: int = 8
    embedding_cache_dir: str = "data/embeddings"

    # ---- prune stage -----------------------------------------------------
    # Hard cap on neighborhood size shipped to the LLM. 0 disables the cap.
    max_subgraph_nodes: int = 200


def _coerce_cap(value: Any, default: int) -> int:
    """``None`` / non-positive → 0 (disabled). Otherwise int-cast."""
    if value is None:
        return 0
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return n if n > 0 else 0


def _section_getter(section: Any) -> Any:
    """Return a ``.get(key, default)`` callable that tolerates missing
    sections (``None``) and non-mapping objects."""

    def _get(key: str, default: Any) -> Any:
        if section is None:
            return default
        try:
            return section.get(key, default)
        except AttributeError:
            return default

    return _get


def build_runner(cfg: Any | None = None):
    """Construct ``AnchorNeighborsRunner`` from the project YAML config."""
    # Local imports to avoid cycles when the registry is imported elsewhere.
    from src.core.config import load_config
    from src.approaches.anchor_neighbors.runner import AnchorNeighborsRunner

    if cfg is None:
        cfg = load_config("configs/config.yaml")

    emb_raw = cfg.get("embeddings") if hasattr(cfg, "get") else None
    approach_raw = cfg.get("approaches") if hasattr(cfg, "get") else None
    section_raw = approach_raw.get("anchor_neighbors") if approach_raw else None

    emb = _section_getter(emb_raw)
    approach = _section_getter(section_raw)
    llm_cfg = cfg.llm

    cache_dir = emb("cache_dir", "data/embeddings") or "data/embeddings"

    runner_cfg = AnchorNeighborsConfig(
        n_candidates=int(approach("n_candidates", 10) or 10),
        embedding_model=emb("model", "BAAI/bge-m3") or "BAAI/bge-m3",
        embedding_device=emb("device", "auto") or "auto",
        embedding_batch_size=int(emb("batch_size", 8) or 8),
        embedding_cache_dir=cache_dir if isinstance(cache_dir, str) else "data/embeddings",
        max_subgraph_nodes=_coerce_cap(approach("max_subgraph_nodes", 200), 200),
    )

    client = LLMClient(
        model=llm_cfg.get("model", "gpt-4-turbo-preview"),
        temperature=llm_cfg.get("temperature", 0.1),
        max_tokens=llm_cfg.get("max_tokens", 4096),
        timeout=llm_cfg.get("timeout", 90),
        retry_attempts=llm_cfg.get("retry_attempts", 3),
        retry_delay=llm_cfg.get("retry_delay", 2),
        api_key=llm_cfg.get("api_key", "") or None,
        base_url=llm_cfg.get("base_url", "") or None,
    )

    return AnchorNeighborsRunner(runner_cfg, client)
