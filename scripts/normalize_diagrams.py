#!/usr/bin/env python3
"""Normalize UML diagrams for the embedding pipeline.

For every JSON diagram in --input-dir (default `uml_with_methods/`):

    1. Drop edge fields other than `node_id_from`, `node_id_to`, `description`.
       The original diagrams carry `subdescription` and `label` (call sites,
       parameter names, multiplicities, etc.) which are noisy and explode the
       graph size; for retrieval-time text we only need the relation kind.

    2. Deduplicate edges. After step 1, many edges collapse to identical
       (from, to, description) triples — keep only one of each.

    3. Preserve nodes verbatim.

The cleaned diagrams are written to --output-dir (default
`data/diagrams_normalized/`), keeping the same file names.

Usage:
    python scripts/normalize_diagrams.py
    python scripts/normalize_diagrams.py --input-dir uml_with_methods --output-dir data/diagrams_normalized
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# Fields kept on every edge after normalization.
EDGE_FIELDS = ("node_id_from", "node_id_to", "description")


def normalize_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip every edge to (from, to, description) and drop duplicates.

    Self-loops (from == to) and edges with empty endpoints are dropped — they
    add no information for retrieval.

    Order is stable: first occurrence wins.
    """
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for e in edges:
        nf = (e.get("node_id_from") or "").strip()
        nt = (e.get("node_id_to") or "").strip()
        desc = (e.get("description") or "").strip()
        if not nf or not nt:
            continue
        if nf == nt:
            continue
        key = (nf, nt, desc)
        if key in seen:
            continue
        seen.add(key)
        out.append({"node_id_from": nf, "node_id_to": nt, "description": desc})
    return out


def normalize_diagram(diagram: dict[str, Any]) -> dict[str, Any]:
    """Return a new diagram dict with normalized edges; nodes are passed through."""
    return {
        "nodes": diagram.get("nodes", []),
        "edges": normalize_edges(diagram.get("edges", [])),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize UML diagrams for RAG.")
    p.add_argument(
        "--input-dir",
        default="uml_with_methods",
        help="Directory containing source JSON diagrams (default: uml_with_methods).",
    )
    p.add_argument(
        "--output-dir",
        default="data/diagrams_normalized",
        help="Where to write the normalized diagrams (default: data/diagrams_normalized).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    if not in_dir.exists():
        print(f"[error] input dir not found: {in_dir}", file=sys.stderr)
        sys.exit(2)

    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(in_dir.glob("*.json"))
    if not paths:
        print(f"[error] no JSON files found in {in_dir}", file=sys.stderr)
        sys.exit(2)

    total_nodes = 0
    total_edges_before = 0
    total_edges_after = 0

    for src in paths:
        dst = out_dir / src.name
        if dst.exists() and not args.overwrite:
            print(f"[skip] {dst} exists (use --overwrite)")
            continue

        with src.open("r", encoding="utf-8") as f:
            diagram = json.load(f)

        edges_before = len(diagram.get("edges", []))
        normalized = normalize_diagram(diagram)
        edges_after = len(normalized["edges"])
        n_nodes = len(normalized["nodes"])

        with dst.open("w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)

        total_nodes += n_nodes
        total_edges_before += edges_before
        total_edges_after += edges_after

        ratio = (edges_after / edges_before * 100.0) if edges_before else 100.0
        print(
            f"{src.name:25s}  nodes={n_nodes:6d}  "
            f"edges {edges_before:7d} -> {edges_after:6d} ({ratio:5.1f}%)"
        )

    if total_edges_before:
        ratio = total_edges_after / total_edges_before * 100.0
    else:
        ratio = 100.0
    print(
        f"\nTOTAL: {len(paths)} diagrams, {total_nodes} nodes, "
        f"{total_edges_before} -> {total_edges_after} edges ({ratio:.1f}% kept)"
    )


if __name__ == "__main__":
    main()
