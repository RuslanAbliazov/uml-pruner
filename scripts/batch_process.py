#!/usr/bin/env python3
"""CLI: run the pipeline on every sample in the consolidated dataset.

Consumes the dataset CSV produced by `scripts/build_dataset.py` (default:
`data/dataset.csv`). For each sample, looks up the diagram via the row's
`repo` column, runs the pipeline with the sample's query, and writes
'{sample_id}.json' to results_dir.

Skips samples whose result file already exists (unless --overwrite).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from tqdm import tqdm

from src.evaluation.annotations import diagram_filename_for_repo, load_dataset
from src.llm.client import LLMClient
from src.llm.prompts import set_prompts_dir
from src.pipeline.pipeline import EmbeddingRetrievalConfig, PipelineConfig, run_pipeline
from src.utils.config import load_config
from src.utils.io import load_diagram, save_diagram
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run the pruning pipeline.")
    parser.add_argument(
        "--dataset",
        default="data/dataset.csv",
        help="Path to the consolidated dataset CSV (built by build_dataset.py).",
    )
    parser.add_argument(
        "--diagrams-dir",
        default="full_diagrams_fixed_generic",
        help="Directory with project diagrams.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/results",
        help="Where to write per-sample result files.",
    )
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Pipeline YAML config."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing result files."
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Process at most N samples (0 = all)."
    )
    parser.add_argument(
        "--repo",
        default="",
        help="Only process samples for this repo slug (e.g. 'apache/hadoop').",
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
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


async def process_all(args: argparse.Namespace) -> None:
    load_dotenv()
    cfg = load_config(args.config)
    setup_logger(
        level="DEBUG" if args.verbose else cfg.logging.get("level", "INFO"),
        log_file=cfg.logging.get("file", None),
    )

    prompts_dir = (
        cfg.paths.get("prompts_dir", "prompts") if hasattr(cfg, "paths") else "prompts"
    )
    set_prompts_dir(prompts_dir)

    samples = load_dataset(args.dataset)
    if args.repo:
        samples = [s for s in samples if s.repo == args.repo]
    if args.limit:
        samples = samples[: args.limit]

    diagrams_dir = Path(args.diagrams_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    # Embedding retrieval config (opt-in). diagram_name is set per-sample below.
    emb_raw = cfg.get("embeddings") if hasattr(cfg, "get") else None
    emb_enabled_from_cfg = bool(emb_raw.get("enabled", False)) if emb_raw else False
    if args.use_embeddings:
        emb_enabled = True
    elif args.no_embeddings:
        emb_enabled = False
    else:
        emb_enabled = emb_enabled_from_cfg

    def _make_embeddings_cfg(diagram_name: str) -> EmbeddingRetrievalConfig:
        return EmbeddingRetrievalConfig(
            enabled=emb_enabled,
            model=(emb_raw.get("model") if emb_raw else None)
            or "nomic-ai/nomic-embed-text-v1.5",
            device=(emb_raw.get("device") if emb_raw else None) or "auto",
            batch_size=(emb_raw.get("batch_size") if emb_raw else None) or 32,
            top_k=(emb_raw.get("top_k") if emb_raw else None) or 300,
            cache_dir=(emb_raw.get("cache_dir") if emb_raw else None)
            or "data/embeddings",
            diagram_name=diagram_name,
            max_methods_per_node=(
                emb_raw.get("max_methods_per_node") if emb_raw else None
            )
            or 15,
            max_description_chars=(
                emb_raw.get("max_description_chars") if emb_raw else None
            )
            or 500,
        )

    def _make_pipeline_cfg(diagram_name: str) -> PipelineConfig:
        return PipelineConfig(
            stage1_batch_size=cfg.pipeline.stage1.get("package_batch_size", 40),
            stage1_parallel=cfg.pipeline.stage1.get("max_parallel_requests", 5),
            stage2_batch_size=cfg.pipeline.stage2.get("class_batch_size", 120),
            stage2_parallel=cfg.pipeline.stage2.get("max_parallel_requests", 3),
            stage2_max_output=cfg.pipeline.stage2.get("max_output_classes", 500),
            context_window=llm_cfg.get("context_window", 128_000),
            output_reserve=llm_cfg.get("output_reserve", 4_096),
            safety_margin=llm_cfg.get("safety_margin", 2_000),
            max_split_depth=cfg.pipeline.get("max_split_depth", 8),
            embeddings=_make_embeddings_cfg(diagram_name),
        )

    diagram_cache: dict[str, dict] = {}

    for sample in tqdm(samples, desc="Processing samples"):
        out_file = output_dir / f"{sample.sample_id}.json"
        if out_file.exists() and not args.overwrite:
            continue

        fname = diagram_filename_for_repo(sample.repo)
        if not fname:
            print(
                f"[skip] Unknown repo for {sample.sample_id}: {sample.repo}"
            )
            continue

        diagram_path = diagrams_dir / fname
        if not diagram_path.exists():
            print(f"[skip] Diagram not found: {diagram_path}")
            continue

        if fname not in diagram_cache:
            diagram_cache[fname] = load_diagram(diagram_path)
        diagram = diagram_cache[fname]

        # diagram_name = filename stem (e.g. "ghidra") — used to locate the
        # embedding index on disk.
        pipeline_cfg = _make_pipeline_cfg(Path(fname).stem)

        try:
            result = await run_pipeline(sample.query, diagram, client, pipeline_cfg)
        except Exception as e:
            print(f"[error] {sample.sample_id}: {e}")
            continue

        # Attach annotation-tracking metadata (doesn't affect evaluation)
        result.setdefault("metadata", {})["sample_id"] = sample.sample_id
        result["metadata"]["repo"] = sample.repo
        save_diagram(result, out_file)

    usage = client.usage_summary()
    print(
        f"\nLLM usage: {usage['total_calls']} calls, "
        f"{usage['input_tokens']:,} input / {usage['output_tokens']:,} output tokens"
    )


def main() -> None:
    args = parse_args()
    asyncio.run(process_all(args))


if __name__ == "__main__":
    main()
