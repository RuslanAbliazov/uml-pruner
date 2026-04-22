#!/usr/bin/env python3
"""Run ONLY the embedding retriever and dump top-K classes to a JSON file.

No LLM calls. Useful for manually inspecting what the retriever returns for
a given query and checking recall against ground-truth annotations.

Prerequisites:
    - Embedding deps installed:   pip install -r requirements-embeddings.txt
    - Index built for the diagram: python scripts/build_index.py --diagram <path>

Usage:
    python scripts/retrieve.py \\
        --diagram full_diagrams_fixed_generic/ghidra.json \\
        --query "Show classes responsible for defining external locations" \\
        --output data/results/ghidra_retrieved.json

    # Override top-K from config:
    python scripts/retrieve.py --diagram ... --query ... --output ... --top-k 500

Output JSON shape:
    {
      "query": "...",
      "diagram": "ghidra",
      "model": "nomic-ai/nomic-embed-text-v1.5",
      "top_k": 300,
      "total_nodes_in_diagram": 19820,
      "retrieved_count": 300,
      "node_ids": ["...", "...", ...],           # ordered by descending score
      "results": [                                # detailed, same order
        {"rank": 1, "node_id": "...", "score": 0.812},
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.cache import compute_diagram_hash, is_valid, load_cache
from src.embeddings.encoder import EncoderConfig, LocalEncoder
from src.embeddings.retriever import retrieve_top_k
from src.utils.config import load_config
from src.utils.io import load_diagram, save_json
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the embedding retriever and save top-K classes to JSON."
    )
    parser.add_argument(
        "--diagram", required=True, help="Path to the UML JSON diagram."
    )
    parser.add_argument("--query", required=True, help="User query (natural language).")
    parser.add_argument(
        "--output", required=True, help="Path to write the retrieved top-K JSON."
    )
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to YAML config file."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override embeddings.top_k from config.",
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
        "--cache-dir",
        default=None,
        help="Override embeddings.cache_dir from config.",
    )
    parser.add_argument(
        "--ignore-hash",
        action="store_true",
        help="Skip diagram-hash validation (use an index even if it looks stale).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(level="DEBUG" if args.verbose else "INFO")
    logger = setup_logger()

    cfg = load_config(args.config)
    emb_raw = cfg.get("embeddings") if hasattr(cfg, "get") else None

    model_name = (
        args.model
        or (emb_raw.get("model") if emb_raw else None)
        or "nomic-ai/nomic-embed-text-v1.5"
    )
    device = args.device or (emb_raw.get("device") if emb_raw else None) or "auto"
    top_k = args.top_k or (emb_raw.get("top_k") if emb_raw else None) or 300
    cache_dir = Path(
        args.cache_dir
        or (emb_raw.get("cache_dir") if emb_raw else None)
        or "data/embeddings"
    )
    batch_size = (emb_raw.get("batch_size") if emb_raw else None) or 32

    diagram_path = Path(args.diagram)
    if not diagram_path.exists():
        logger.error("Diagram not found: %s", diagram_path)
        sys.exit(1)
    diagram_name = diagram_path.stem

    # Load diagram (only for metadata + optional hash validation)
    diagram = load_diagram(diagram_path)
    logger.info(
        "Diagram '%s' loaded: %d nodes, %d edges",
        diagram_name,
        len(diagram["nodes"]),
        len(diagram["edges"]),
    )

    # Load index
    entry = load_cache(cache_dir, diagram_name)
    if entry is None:
        logger.error(
            "No embedding index found at %s/%s. "
            "Build one with: python scripts/build_index.py --diagram %s",
            cache_dir,
            diagram_name,
            diagram_path,
        )
        sys.exit(2)

    # Validate index (skippable with --ignore-hash)
    if not args.ignore_hash:
        expected_hash = compute_diagram_hash(diagram["nodes"])
        if not is_valid(
            entry, expected_model=model_name, expected_diagram_hash=expected_hash
        ):
            logger.error(
                "Index at %s/%s is stale (model or diagram changed). "
                "Rebuild with: python scripts/build_index.py --diagram %s --force  "
                "(or pass --ignore-hash to proceed anyway)",
                cache_dir,
                diagram_name,
                diagram_path,
            )
            sys.exit(3)
    else:
        logger.warning("--ignore-hash: skipping index validation")

    # Encode + retrieve
    encoder = LocalEncoder(
        EncoderConfig(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )
    )
    logger.info("Retrieving top-%d for query: %r", top_k, args.query)
    hits = retrieve_top_k(args.query, entry, encoder, top_k=top_k)

    # Build output
    output = {
        "query": args.query,
        "diagram": diagram_name,
        "diagram_path": str(diagram_path),
        "model": model_name,
        "device": encoder.device,
        "top_k": top_k,
        "total_nodes_in_diagram": len(diagram["nodes"]),
        "total_nodes_indexed": len(entry),
        "retrieved_count": len(hits),
        "node_ids": [h.node_id for h in hits],
        "results": [
            {"rank": i + 1, "node_id": h.node_id, "score": round(h.score, 6)}
            for i, h in enumerate(hits)
        ],
    }

    save_json(output, args.output)
    logger.info(
        "Wrote %d results to %s (score range: %.3f .. %.3f)",
        len(hits),
        args.output,
        hits[-1].score if hits else 0.0,
        hits[0].score if hits else 0.0,
    )


if __name__ == "__main__":
    main()
