#!/usr/bin/env python3
"""Build an embedding index for one or more UML diagrams.

Usage:
    # Build index for a single diagram:
    python scripts/build_index.py --diagram full_diagrams_fixed_generic/disruptor.json

    # Build indices for all diagrams in a directory:
    python scripts/build_index.py --all

    # Override model / device from the command line:
    python scripts/build_index.py --diagram <path> --model BAAI/bge-small-en-v1.5 --device cpu

    # Force rebuild even if a valid cache exists:
    python scripts/build_index.py --all --force

The index is stored under `<cache_dir>/<diagram_name>/` where <diagram_name>
is the JSON file's stem. See src/embeddings/cache.py for the layout.

This script is INDEPENDENT of the LLM pipeline: no OpenAI key required.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.cache import (
    compute_diagram_hash,
    is_valid,
    load_cache,
    save_cache,
)
from src.embeddings.encoder import EncoderConfig, LocalEncoder
from src.embeddings.node_to_text import nodes_to_texts
from src.utils.config import load_config
from src.utils.io import load_diagram
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build embedding indices for UML diagrams."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--diagram", help="Path to a single diagram JSON file.")
    group.add_argument(
        "--all",
        action="store_true",
        help="Build indices for every *.json in the diagrams directory.",
    )
    parser.add_argument(
        "--diagrams-dir",
        default=None,
        help="Directory with diagrams (defaults to paths.diagrams_dir from config).",
    )
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to YAML config file."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override embeddings.model from config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override embeddings.device from config (auto|cuda|mps|cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override embeddings.batch_size from config.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Override embeddings.cache_dir from config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing cache and re-embed from scratch.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _resolve_diagrams(args: argparse.Namespace, diagrams_dir: Path) -> list[Path]:
    if args.all:
        return sorted(diagrams_dir.glob("*.json"))
    return [Path(args.diagram)]


def _index_one(
    diagram_path: Path,
    encoder: LocalEncoder,
    cache_dir: Path,
    text_opts: dict,
    force: bool,
    logger,
) -> None:
    diagram_name = diagram_path.stem
    logger.info("=== %s ===", diagram_name)

    diagram = load_diagram(diagram_path)
    nodes = diagram["nodes"]
    logger.info("Loaded: %d nodes, %d edges", len(nodes), len(diagram["edges"]))

    diagram_hash = compute_diagram_hash(nodes)

    if not force:
        existing = load_cache(cache_dir, diagram_name)
        if existing and is_valid(existing, encoder.model_name, diagram_hash):
            logger.info(
                "Valid cache already exists (%d vectors x %d dim). "
                "Use --force to rebuild.",
                len(existing),
                existing.dimension,
            )
            return

    # Build texts
    logger.info("Serializing %d nodes to text...", len(nodes))
    texts = nodes_to_texts(
        nodes,
        max_methods=text_opts["max_methods"],
        max_description_chars=text_opts["max_description_chars"],
    )
    node_ids = [n["node_id"] for n in nodes]

    # Encode
    logger.info(
        "Encoding with model='%s' device='%s' batch_size=%d",
        encoder.model_name,
        encoder.device,
        encoder.cfg.batch_size,
    )
    t0 = time.time()
    vectors = encoder.encode_documents(texts, show_progress_bar=True)
    elapsed = time.time() - t0
    logger.info(
        "Encoded %d nodes in %.1f s (%.1f nodes/s)",
        len(node_ids),
        elapsed,
        len(node_ids) / max(elapsed, 0.001),
    )

    # Save
    save_cache(
        base_dir=cache_dir,
        diagram_name=diagram_name,
        model_name=encoder.model_name,
        node_ids=node_ids,
        vectors=vectors,
        diagram_hash=diagram_hash,
        extra_meta={
            "encoder": encoder.describe(),
            "text_opts": text_opts,
        },
    )


def main() -> None:
    args = parse_args()
    setup_logger(level="DEBUG" if args.verbose else "INFO")
    logger = setup_logger()

    cfg = load_config(args.config)
    emb_cfg = cfg.get("embeddings", None)

    # Merge config + CLI overrides
    model_name = (
        args.model
        or (emb_cfg.get("model") if emb_cfg else None)
        or "nomic-ai/nomic-embed-text-v1.5"
    )
    device = args.device or (emb_cfg.get("device") if emb_cfg else None) or "auto"
    batch_size = (
        args.batch_size or (emb_cfg.get("batch_size") if emb_cfg else None) or 32
    )
    cache_dir = Path(
        args.cache_dir
        or (emb_cfg.get("cache_dir") if emb_cfg else None)
        or "data/embeddings"
    )
    text_opts = {
        "max_methods": (emb_cfg.get("max_methods_per_node") if emb_cfg else None) or 15,
        "max_description_chars": (
            emb_cfg.get("max_description_chars") if emb_cfg else None
        )
        or 500,
    }

    diagrams_dir = Path(
        args.diagrams_dir
        or (cfg.paths.get("diagrams_dir") if hasattr(cfg, "paths") else None)
        or "diagrams"
    )

    diagrams = _resolve_diagrams(args, diagrams_dir)
    if not diagrams:
        logger.error("No diagrams to process.")
        sys.exit(1)

    encoder = LocalEncoder(
        EncoderConfig(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    for path in diagrams:
        if not path.exists():
            logger.error("Diagram not found: %s", path)
            continue
        _index_one(path, encoder, cache_dir, text_opts, args.force, logger)

    logger.info("Done.")


if __name__ == "__main__":
    main()
