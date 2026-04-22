"""Main pipeline orchestrator (2 stages).

Stage 1 (LLM, default) OR Embedding Retrieval (optional, opt-in):
    - LLM Stage 1: groups classes by package, asks LLM which packages are
      relevant (see src/pipeline/stage1_coarse.py).
    - Embedding retrieval: loads a pre-computed vector index and returns the
      top-K classes most similar to the query (see src/embeddings/*).

Stage 2 (always LLM):
    For each candidate class, LLM classifies it as REQUIRED / USEFUL / IRRELEVANT.

Both stages are protected from context overflow by the autosplit driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.llm.budget import TokenBudget
from src.llm.client import LLMClient
from src.pipeline.stage1_coarse import run_stage1
from src.pipeline.stage2_midlevel import run_stage2
from src.preprocessing.compressor import filter_subgraph
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingRetrievalConfig:
    """Opt-in embedding-based replacement for Stage 1."""

    enabled: bool = False
    model: str = "nomic-ai/nomic-embed-text-v1.5"
    device: str = "auto"
    batch_size: int = 32
    top_k: int = 300
    cache_dir: str = "data/embeddings"
    # Diagram identifier (stem of the diagram file). If not provided, the
    # caller should set this before invoking the pipeline so the retriever
    # can locate the right index on disk.
    diagram_name: str = ""
    max_methods_per_node: int = 15
    max_description_chars: int = 500


@dataclass
class PipelineConfig:
    # Stage 1 (package filtering, LLM — used when embeddings disabled)
    stage1_batch_size: int = 40
    stage1_parallel: int = 5
    # Stage 2 (class filtering, final)
    stage2_batch_size: int = 120
    stage2_parallel: int = 3
    stage2_max_output: int = 500
    # Overflow protection (token budget, shared by both stages)
    context_window: int = 128_000
    output_reserve: int = 4_096
    safety_margin: int = 2_000
    max_split_depth: int = 8
    # Embedding retrieval (opt-in)
    embeddings: EmbeddingRetrievalConfig = field(
        default_factory=EmbeddingRetrievalConfig
    )


def _build_budget(cfg: PipelineConfig) -> TokenBudget:
    return TokenBudget(
        context_window=cfg.context_window,
        output_reserve=cfg.output_reserve,
        safety_margin=cfg.safety_margin,
    )


async def _stage1_via_embeddings(
    query: str,
    nodes: list[dict[str, Any]],
    cfg: PipelineConfig,
) -> tuple[set[str], str]:
    """Use a pre-computed embedding index for Stage 1.

    Returns:
        (set of surviving node_ids, source identifier string).

    If the index is missing or embedding deps are not installed, returns an
    empty set and a diagnostic source string so the caller can fall back to
    LLM Stage 1.
    """
    ecfg = cfg.embeddings
    diagram_name = ecfg.diagram_name
    if not diagram_name:
        logger.warning(
            "Embeddings enabled but diagram_name is empty; falling back to LLM Stage 1"
        )
        return set(), "embeddings_no_diagram_name"

    try:
        from src.embeddings.cache import (
            compute_diagram_hash,
            is_valid,
            load_cache,
        )
        from src.embeddings.encoder import EncoderConfig, LocalEncoder
        from src.embeddings.retriever import retrieve_top_k_ids
    except ImportError as e:
        logger.warning(
            "Embedding deps not installed (%s); falling back to LLM Stage 1."
            " Install with: pip install -r requirements-embeddings.txt",
            e,
        )
        return set(), "embeddings_deps_missing"

    entry = load_cache(ecfg.cache_dir, diagram_name)
    if entry is None:
        logger.warning(
            "No embedding index found at %s/%s — falling back to LLM Stage 1. "
            "Build one via: python scripts/build_index.py --diagram <path>",
            ecfg.cache_dir,
            diagram_name,
        )
        return set(), "embeddings_no_index"

    # Validate model + diagram hash
    expected_hash = compute_diagram_hash(nodes)
    if not is_valid(
        entry, expected_model=ecfg.model, expected_diagram_hash=expected_hash
    ):
        logger.warning(
            "Embedding index at %s/%s is stale (model or diagram changed); "
            "falling back to LLM Stage 1. Rebuild with --force.",
            ecfg.cache_dir,
            diagram_name,
        )
        return set(), "embeddings_stale_index"

    # Run retrieval
    encoder = LocalEncoder(
        EncoderConfig(
            model_name=ecfg.model,
            device=ecfg.device,
            batch_size=ecfg.batch_size,
        )
    )
    logger.info(
        "Using embedding retrieval: index=%s, model=%s, top_k=%d",
        diagram_name,
        ecfg.model,
        ecfg.top_k,
    )
    ids = retrieve_top_k_ids(query, entry, encoder, top_k=ecfg.top_k)

    # Defensive: only keep ids that actually exist in the current diagram
    valid_ids = {n["node_id"] for n in nodes}
    filtered = ids & valid_ids
    logger.info(
        "Embedding retrieval: %d hits kept (out of %d from index, %d in diagram)",
        len(filtered),
        len(ids),
        len(valid_ids),
    )
    if not filtered:
        logger.warning(
            "Embedding retrieval returned 0 valid nodes; falling back to LLM Stage 1"
        )
        return set(), "embeddings_empty_result"

    return filtered, "embeddings"


async def run_pipeline(
    query: str,
    diagram: dict[str, Any],
    llm_client: LLMClient,
    cfg: PipelineConfig | None = None,
) -> dict[str, Any]:
    """Run the 2-stage pipeline.

    Returns the pruned diagram in the original {nodes, edges} shape, plus a
    metadata block with diagnostic info.
    """
    cfg = cfg or PipelineConfig()
    budget = _build_budget(cfg)

    nodes = diagram["nodes"]
    edges = diagram["edges"]
    original_count = len(nodes)

    # -- Stage 1: filter packages (LLM) or embedding retrieval --------------
    stage1_source = "llm"
    stage1_ids: set[str] = set()

    if cfg.embeddings.enabled:
        stage1_ids, src = await _stage1_via_embeddings(query, nodes, cfg)
        stage1_source = src

    if not stage1_ids:
        # Fallback to LLM Stage 1 (also the default when embeddings disabled).
        stage1_ids = await run_stage1(
            query,
            nodes,
            llm_client,
            budget,
            batch_size=cfg.stage1_batch_size,
            max_parallel=cfg.stage1_parallel,
            max_split_depth=cfg.max_split_depth,
        )
        if stage1_source != "llm":
            # We tried embeddings but fell back
            stage1_source = f"llm_after_{stage1_source}"

    # -- Stage 2: classify remaining classes as REQUIRED/USEFUL/IRRELEVANT --
    stage2_result = await run_stage2(
        query,
        nodes,
        edges,
        stage1_ids,
        llm_client,
        budget,
        batch_size=cfg.stage2_batch_size,
        max_parallel=cfg.stage2_parallel,
        max_output=cfg.stage2_max_output,
        max_split_depth=cfg.max_split_depth,
    )

    # -- Assemble final output --------------------------------------------
    final_ids = stage2_result.all_kept
    final_nodes, final_edges = filter_subgraph(nodes, edges, final_ids)
    reduction_ratio = len(final_nodes) / original_count if original_count else 0.0

    result: dict[str, Any] = {
        "nodes": final_nodes,
        "edges": final_edges,
        "metadata": {
            "query": query,
            "original_node_count": original_count,
            "filtered_node_count": len(final_nodes),
            "reduction_ratio": round(reduction_ratio, 6),
            "stage1_source": stage1_source,
            "stage_sizes": {
                "stage1_survivors": len(stage1_ids),
                "stage2_required": len(stage2_result.required),
                "stage2_useful": len(stage2_result.useful),
                "stage2_total": len(final_ids),
            },
            "required_node_ids": sorted(stage2_result.required),
            "useful_node_ids": sorted(stage2_result.useful),
        },
    }

    logger.info(
        "Pipeline done: %d -> %d (%.2f%%), stage1=%s, required=%d useful=%d",
        original_count,
        len(final_nodes),
        100.0 * reduction_ratio,
        stage1_source,
        len(stage2_result.required),
        len(stage2_result.useful),
    )
    return result
