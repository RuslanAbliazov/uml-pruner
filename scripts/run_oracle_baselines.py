#!/usr/bin/env python3
"""Run oracle baselines that intentionally see ground truth.

Oracle baselines are NOT real approaches — they look at the central_node
and/or annotations, which the production pipeline must never see. They
exist to give us upper bounds and sanity checks.

Each baseline writes results into the standard
``data/results/<name>/<sample_id>.json`` shape, so the standard
``scripts/eval.py`` and ``src.eval.evaluator`` pick them up unchanged.

Usage:
    # Run both, evaluate, write reports:
    python scripts/run_oracle_baselines.py

    # Pick specific baselines:
    python scripts/run_oracle_baselines.py --baselines central_plus_neighbors

    # Restrict to one repo / one sample:
    python scripts/run_oracle_baselines.py --repo apache/hadoop
    python scripts/run_oracle_baselines.py --sample-id <id>

    # Custom paths:
    python scripts/run_oracle_baselines.py \
        --dataset      data/dataset.csv \
        --diagrams-dir data/diagrams_normalized \
        --output-root  data/results

Output goes to ``<output_root>/oracle_<baseline_name>/<sample_id>.json``,
plus an ``evaluation_report.json`` per baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.io import load_diagram, save_diagram, save_json  # noqa: E402
from src.eval.annotations import diagram_filename_for_repo, load_dataset  # noqa: E402
from src.eval.evaluator import evaluate_test_set, format_summary_report  # noqa: E402
from src.eval.oracle_baselines import ORACLES  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run oracle baselines.")
    p.add_argument(
        "--baselines",
        nargs="+",
        default=sorted(ORACLES.keys()),
        choices=sorted(ORACLES.keys()),
        help=f"Which oracles to run (default: all). Available: {sorted(ORACLES.keys())}",
    )
    p.add_argument("--dataset", default="data/dataset.csv")
    p.add_argument("--diagrams-dir", default="data/diagrams_normalized")
    p.add_argument("--output-root", default="data/results")
    p.add_argument("--repo", default="")
    p.add_argument("--sample-id", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-eval", action="store_true")
    return p.parse_args()


def _filter_subgraph(
    nodes: list[dict], edges: list[dict], keep: set[str]
) -> tuple[list[dict], list[dict]]:
    nodes_out = [n for n in nodes if n.get("node_id") in keep]
    edges_out = [
        e
        for e in edges
        if e.get("node_id_from") in keep and e.get("node_id_to") in keep
    ]
    return nodes_out, edges_out


def main() -> None:
    args = parse_args()

    samples = load_dataset(args.dataset)
    if args.repo:
        samples = [s for s in samples if s.repo == args.repo]
    if args.sample_id:
        samples = [s for s in samples if s.sample_id == args.sample_id]
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        print("[error] no samples to run after filtering.", file=sys.stderr)
        sys.exit(2)

    diagrams_dir = Path(args.diagrams_dir)
    output_root = Path(args.output_root)
    diagram_cache: dict[str, dict] = {}

    for baseline_name in args.baselines:
        predict = ORACLES[baseline_name]
        out_dir = output_root / f"oracle_{baseline_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        n_done = 0
        n_skipped = 0
        for sample in samples:
            fname = diagram_filename_for_repo(sample.repo)
            if fname not in diagram_cache:
                path = diagrams_dir / fname
                if not path.exists():
                    n_skipped += 1
                    continue
                diagram_cache[fname] = load_diagram(path)
            diagram = diagram_cache[fname]

            keep = predict(sample, diagram)
            nodes_out, edges_out = _filter_subgraph(
                diagram.get("nodes", []), diagram.get("edges", []), keep
            )

            out = {
                "nodes": nodes_out,
                "edges": edges_out,
                "metadata": {
                    "approach": f"oracle_{baseline_name}",
                    "sample_id": sample.sample_id,
                    "repo": sample.repo,
                    "query": sample.query,
                    "central_node": sample.central_node,
                    "n_predicted": len(keep),
                    "is_oracle": True,
                    "warning": (
                        "Oracle baseline — reads ground truth. "
                        "Do NOT report as a real approach result."
                    ),
                },
            }
            save_diagram(out, out_dir / f"{sample.sample_id}.json")
            n_done += 1

        print(
            f"[{baseline_name}] wrote {n_done} sample(s), "
            f"{n_skipped} skipped (missing diagram). "
            f"Output: {out_dir}"
        )

        if not args.no_eval:
            result = evaluate_test_set(args.dataset, str(out_dir))
            print(format_summary_report(result))
            save_json(result.to_dict(), out_dir / "evaluation_report.json")


if __name__ == "__main__":
    main()
