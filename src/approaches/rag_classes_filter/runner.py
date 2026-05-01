"""Approach #1: RAG-batched candidate set + LLM classifier (baseline).

Pipeline shape:

    1. Stage 1 — embedding retrieval (preferred) OR LLM package filter:
       grab a batch of 50–200 candidate classes that look topically related
       to the query.
    2. Stage 2 — LLM classifier: send the candidate batch (with their
       neighborhood context) to the LLM and ask which classes are REQUIRED,
       USEFUL or IRRELEVANT.

This is the legacy 2-stage pipeline already implemented in
``src/pipeline/pipeline.py``; here we just adapt it to the
:class:`ApproachRunner` interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.types import ApproachInputs, ApproachResult
from src.llm.client import LLMClient
from src.approaches.rag_classes_filter.pipeline import (
    EmbeddingRetrievalConfig,
    PipelineConfig,
    run_pipeline,
)
from src.core.logger import get_logger

NAME = "rag_classes_filter"

logger = get_logger(__name__)


@dataclass
class RagClassesFilterConfig:
    """Knobs for approach #1.

    Built from the YAML config by :func:`build_runner`. All defaults match the
    project's existing 2-stage pipeline behavior.
    """

    pipeline: PipelineConfig
    llm_client: LLMClient


class RagClassesFilterRunner:
    """Adapter around :func:`src.approaches.rag_classes_filter.pipeline.run_pipeline`."""

    name = NAME

    def __init__(self, cfg: RagClassesFilterConfig) -> None:
        self._cfg = cfg

    async def run(self, inputs: ApproachInputs) -> ApproachResult:
        # Tell the embedding stage which on-disk index to load.
        if self._cfg.pipeline.embeddings.enabled and inputs.repo:
            self._cfg.pipeline.embeddings.diagram_name = _diagram_name_for_repo(
                inputs.repo
            )

        result = await run_pipeline(
            inputs.query, inputs.diagram, self._cfg.llm_client, self._cfg.pipeline
        )

        meta = dict(result.get("metadata") or {})
        return ApproachResult(
            approach=self.name,
            nodes=result.get("nodes", []),
            edges=result.get("edges", []),
            required_node_ids=list(meta.get("required_node_ids") or []),
            useful_node_ids=list(meta.get("useful_node_ids") or []),
            metadata=meta,
        )

    async def aclose(self) -> None:
        # LLMClient does not own a long-lived connection; nothing to close.
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _diagram_name_for_repo(repo: str) -> str:
    """Convert a repo slug ('apache/hadoop') to its diagram-file stem.

    Mirrors the logic in :mod:`src.eval.annotations` but kept private to
    avoid a circular dependency.
    """
    if "/" in repo:
        repo = repo.split("/", 1)[1]
    return Path(repo).stem


def build_runner(cfg: Any | None = None) -> RagClassesFilterRunner:
    """Build the runner from a project YAML config (``ConfigDict``).

    Falls back to sensible defaults if ``cfg`` is None — useful for tests.
    """
    from src.core.config import load_config  # local import to avoid cycles

    if cfg is None:
        cfg = load_config("configs/config.yaml")

    llm_cfg = cfg.llm
    pipeline_yaml = cfg.pipeline
    emb_raw = cfg.get("embeddings") if hasattr(cfg, "get") else None

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

    embeddings_cfg = EmbeddingRetrievalConfig(
        enabled=bool(emb_raw.get("enabled", False)) if emb_raw else False,
        model=(emb_raw.get("model") if emb_raw else None)
        or "nomic-ai/nomic-embed-text-v1.5",
        device=(emb_raw.get("device") if emb_raw else None) or "auto",
        batch_size=(emb_raw.get("batch_size") if emb_raw else None) or 32,
        top_k=(emb_raw.get("top_k") if emb_raw else None) or 300,
        cache_dir=(emb_raw.get("cache_dir") if emb_raw else None) or "data/embeddings",
        diagram_name="",  # set per-sample in run()
        max_methods_per_node=(emb_raw.get("max_methods_per_node") if emb_raw else None)
        or 25,
        max_description_chars=(
            emb_raw.get("max_description_chars") if emb_raw else None
        )
        or 500,
    )

    pipeline_cfg = PipelineConfig(
        stage1_batch_size=pipeline_yaml.stage1.get("package_batch_size", 40),
        stage1_parallel=pipeline_yaml.stage1.get("max_parallel_requests", 5),
        stage2_batch_size=pipeline_yaml.stage2.get("class_batch_size", 120),
        stage2_parallel=pipeline_yaml.stage2.get("max_parallel_requests", 3),
        stage2_max_output=pipeline_yaml.stage2.get("max_output_classes", 500),
        context_window=llm_cfg.get("context_window", 128_000),
        output_reserve=llm_cfg.get("output_reserve", 4_096),
        safety_margin=llm_cfg.get("safety_margin", 2_000),
        max_split_depth=pipeline_yaml.get("max_split_depth", 8),
        embeddings=embeddings_cfg,
    )

    return RagClassesFilterRunner(
        RagClassesFilterConfig(pipeline=pipeline_cfg, llm_client=client)
    )
