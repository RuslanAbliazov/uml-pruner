#!/usr/bin/env python3
"""CLI: run the pruning pipeline on a single diagram + query."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.llm.client import LLMClient
from src.llm.prompts import set_prompts_dir
from src.pipeline.pipeline import EmbeddingRetrievalConfig, PipelineConfig, run_pipeline
from src.utils.config import load_config
from src.utils.io import load_diagram, save_diagram
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune a UML diagram using an LLM pipeline."
    )
    parser.add_argument("--query", required=True, help="User query (natural language).")
    parser.add_argument(
        "--diagram", required=True, help="Path to input UML JSON diagram."
    )
    parser.add_argument(
        "--output", required=True, help="Path to write pruned diagram JSON."
    )
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to YAML config file."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG-level logging."
    )
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        help="Override config: force use of embedding-based Stage 1 retrieval.",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Override config: force LLM-based Stage 1 (disable embeddings).",
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    load_dotenv()

    cfg = load_config(args.config)
    log_level = "DEBUG" if args.verbose else cfg.logging.get("level", "INFO")
    logger = setup_logger(
        level=log_level,
        log_file=cfg.logging.get("file", None),
        fmt=cfg.logging.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
    )

    prompts_dir = (
        cfg.paths.get("prompts_dir", "prompts") if hasattr(cfg, "paths") else "prompts"
    )
    set_prompts_dir(prompts_dir)

    logger.info("Loading diagram from %s", args.diagram)
    diagram = load_diagram(args.diagram)
    logger.info(
        "Diagram loaded: %d nodes, %d edges",
        len(diagram["nodes"]),
        len(diagram["edges"]),
    )

    llm_cfg = cfg.llm
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

    # Embedding retrieval config (opt-in)
    emb_raw = cfg.get("embeddings") if hasattr(cfg, "get") else None
    emb_enabled_from_cfg = bool(emb_raw.get("enabled", False)) if emb_raw else False
    if args.use_embeddings:
        emb_enabled = True
    elif args.no_embeddings:
        emb_enabled = False
    else:
        emb_enabled = emb_enabled_from_cfg

    embeddings_cfg = EmbeddingRetrievalConfig(
        enabled=emb_enabled,
        model=(emb_raw.get("model") if emb_raw else None)
        or "nomic-ai/nomic-embed-text-v1.5",
        device=(emb_raw.get("device") if emb_raw else None) or "auto",
        batch_size=(emb_raw.get("batch_size") if emb_raw else None) or 32,
        top_k=(emb_raw.get("top_k") if emb_raw else None) or 300,
        cache_dir=(emb_raw.get("cache_dir") if emb_raw else None) or "data/embeddings",
        diagram_name=Path(args.diagram).stem,
        max_methods_per_node=(emb_raw.get("max_methods_per_node") if emb_raw else None)
        or 15,
        max_description_chars=(
            emb_raw.get("max_description_chars") if emb_raw else None
        )
        or 500,
    )

    pipeline_cfg = PipelineConfig(
        stage1_batch_size=cfg.pipeline.stage1.get("package_batch_size", 40),
        stage1_parallel=cfg.pipeline.stage1.get("max_parallel_requests", 5),
        stage2_batch_size=cfg.pipeline.stage2.get("class_batch_size", 120),
        stage2_parallel=cfg.pipeline.stage2.get("max_parallel_requests", 3),
        stage2_max_output=cfg.pipeline.stage2.get("max_output_classes", 500),
        context_window=llm_cfg.get("context_window", 128_000),
        output_reserve=llm_cfg.get("output_reserve", 4_096),
        safety_margin=llm_cfg.get("safety_margin", 2_000),
        max_split_depth=cfg.pipeline.get("max_split_depth", 8),
        embeddings=embeddings_cfg,
    )

    result = await run_pipeline(args.query, diagram, client, pipeline_cfg)

    save_diagram(result, args.output)
    logger.info("Pruned diagram written to %s", args.output)

    usage = client.usage_summary()
    logger.info(
        "LLM usage: %d calls, %d input tokens, %d output tokens",
        usage["total_calls"],
        usage["input_tokens"],
        usage["output_tokens"],
    )


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
