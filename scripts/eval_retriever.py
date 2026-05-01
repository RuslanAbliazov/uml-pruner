#!/usr/bin/env python3
"""Evaluate the embedding retriever ALONE (no LLM calls).

For each sample in the consolidated dataset (built by `build_dataset.py`),
runs the embedding retriever against the pre-built index, and measures how
many of the ground-truth 'required' / 'useful' classes end up in the top-K
candidates.

This is the key validation for the embedding-based Stage 1: if recall_required
is ~1.0 at top-K, we know the retriever is not losing critical classes, and
the LLM Stage 2 has a chance to find them.

Usage:
    # Evaluate all samples at the default top-K from config:
    python scripts/eval_retriever.py

    # Sweep multiple K values in one run:
    python scripts/eval_retriever.py --top-k 100 300 500 1000

    # Only one sample, verbose (to inspect misses):
    python scripts/eval_retriever.py --sample-id <id> --show-misses

    # Only one repo:
    python scripts/eval_retriever.py --repo NationalSecurityAgency/ghidra

    # Output JSON report:
    python scripts/eval_retriever.py --output data/results/retriever_eval.json

Requires:
    - The dataset CSV to exist (run `python scripts/build_dataset.py` first).
    - An embedding index to exist for every project being evaluated
      (run `python scripts/build_index.py --all` first).
    - Embedding deps installed (`pip install -r requirements-embeddings.txt`).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.cache import compute_diagram_hash, is_valid, load_cache
from src.rag.encoder import EncoderConfig, LocalEncoder
from src.rag.retriever import retrieve_top_k
from src.eval.annotations import (
    AnnotationSample,
    diagram_filename_for_repo,
    load_dataset,
)
from src.core.config import load_config
from src.core.io import load_diagram, save_json
from src.core.logger import setup_logger

logger = setup_logger()


# -----------------------------------------------------------------------------
# Arg parsing
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the embedding retriever against ground-truth annotations."
    )
    parser.add_argument(
        "--dataset",
        default="data/dataset.csv",
        help="Path to the consolidated dataset CSV (built by build_dataset.py).",
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
        "--top-k",
        type=int,
        nargs="+",
        default=None,
        help="One or more K values to evaluate (default: embeddings.top_k from config).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override embeddings.model from config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override embeddings.device from config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override embeddings.batch_size (for query encoding).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Override embeddings.cache_dir from config.",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Evaluate only the sample with this sample_id.",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Only evaluate samples for this repo slug (e.g. 'apache/hadoop').",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Evaluate at most N samples (0 = all)."
    )
    parser.add_argument(
        "--show-misses",
        action="store_true",
        help="Print which required/useful classes were missed by the retriever.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the full JSON report.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Per-sample recall computation
# -----------------------------------------------------------------------------


def _split_annotations(sample: AnnotationSample) -> tuple[set[str], set[str]]:
    """Return (required_set, useful_set). Irrelevant is ignored."""
    req = {cls for cls, lbl in sample.annotations.items() if lbl == "required"}
    use = {cls for cls, lbl in sample.annotations.items() if lbl == "useful"}
    return req, use


def _recall(retrieved: set[str], truth: set[str]) -> float:
    if not truth:
        return 1.0  # nothing to find -> trivially perfect
    return len(retrieved & truth) / len(truth)


def _eval_sample(
    sample: AnnotationSample,
    hits_ids: set[str],
    k: int,
) -> dict[str, Any]:
    required, useful = _split_annotations(sample)
    retrieved_required = hits_ids & required
    retrieved_useful = hits_ids & useful
    missed_required = sorted(required - hits_ids)
    missed_useful = sorted(useful - hits_ids)
    return {
        "sample_id": sample.sample_id,
        "repo": sample.repo,
        "project": sample.project,  # backwards-compat (diagram stem)
        "k": k,
        "query": sample.query,
        "central_node": sample.central_node,
        "required_total": len(required),
        "useful_total": len(useful),
        "required_found": len(retrieved_required),
        "useful_found": len(retrieved_useful),
        "recall_required": _recall(hits_ids, required),
        "recall_useful": _recall(hits_ids, useful),
        "recall_overall": _recall(hits_ids, required | useful),
        "missed_required": missed_required,
        "missed_useful": missed_useful,
    }


# -----------------------------------------------------------------------------
# Aggregation and reporting
# -----------------------------------------------------------------------------


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-sample metrics (macro averages + micro totals)."""
    if not results:
        return {}

    def mean(key: str) -> float:
        vals = [r[key] for r in results]
        return sum(vals) / len(vals) if vals else 0.0

    # Micro totals
    tot_req = sum(r["required_total"] for r in results)
    tot_req_found = sum(r["required_found"] for r in results)
    tot_use = sum(r["useful_total"] for r in results)
    tot_use_found = sum(r["useful_found"] for r in results)

    micro_req = tot_req_found / tot_req if tot_req else 0.0
    micro_use = tot_use_found / tot_use if tot_use else 0.0
    micro_overall = (
        (tot_req_found + tot_use_found) / (tot_req + tot_use)
        if (tot_req + tot_use)
        else 0.0
    )

    # Samples with perfect required recall
    perfect_required = sum(
        1 for r in results if r["required_total"] == 0 or r["recall_required"] == 1.0
    )

    # Per-project breakdown: how many samples have >=1 required class retrieved
    # (only counting samples that actually have required-labelled classes).
    by_project: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_project[r["project"]].append(r)

    project_any_required: dict[str, dict[str, Any]] = {}
    total_with_req = 0
    total_hit = 0
    total_no_req = 0
    for proj, rows in by_project.items():
        with_req = [r for r in rows if r["required_total"] > 0]
        no_req = len(rows) - len(with_req)
        hit = sum(1 for r in with_req if r["required_found"] >= 1)
        project_any_required[proj] = {
            "samples_total": len(rows),
            "samples_with_required": len(with_req),
            "samples_without_required": no_req,
            "samples_with_any_required_hit": hit,
            "ratio": (hit / len(with_req)) if with_req else 0.0,
        }
        total_with_req += len(with_req)
        total_hit += hit
        total_no_req += no_req

    overall_any_required = {
        "samples_total": len(results),
        "samples_with_required": total_with_req,
        "samples_without_required": total_no_req,
        "samples_with_any_required_hit": total_hit,
        "ratio": (total_hit / total_with_req) if total_with_req else 0.0,
    }

    return {
        "num_samples": len(results),
        "macro_recall_required": mean("recall_required"),
        "macro_recall_useful": mean("recall_useful"),
        "macro_recall_overall": mean("recall_overall"),
        "micro_recall_required": micro_req,
        "micro_recall_useful": micro_use,
        "micro_recall_overall": micro_overall,
        "perfect_required_samples": perfect_required,
        "perfect_required_ratio": perfect_required / len(results),
        "totals": {
            "required": tot_req,
            "required_found": tot_req_found,
            "useful": tot_use,
            "useful_found": tot_use_found,
        },
        "any_required_hit_overall": overall_any_required,
        "any_required_hit_by_project": project_any_required,
    }


def _print_summary(
    k: int, summary: dict[str, Any], per_sample: list[dict[str, Any]]
) -> None:
    print()
    print("=" * 70)
    print(f"RETRIEVER EVALUATION — top_k={k}")
    print("=" * 70)
    n = summary.get("num_samples", 0)
    if not n:
        print("No samples evaluated.")
        return
    print(f"Samples: {n}")
    print()
    print(f"  Macro recall (required):  {summary['macro_recall_required']:.4f}")
    print(f"  Macro recall (useful):    {summary['macro_recall_useful']:.4f}")
    print(f"  Macro recall (overall):   {summary['macro_recall_overall']:.4f}")
    print()
    t = summary["totals"]
    print(
        f"  Micro recall (required):  {summary['micro_recall_required']:.4f} "
        f"({t['required_found']}/{t['required']})"
    )
    print(
        f"  Micro recall (useful):    {summary['micro_recall_useful']:.4f} "
        f"({t['useful_found']}/{t['useful']})"
    )
    print(
        f"  Micro recall (overall):   {summary['micro_recall_overall']:.4f} "
        f"({t['required_found'] + t['useful_found']}/{t['required'] + t['useful']})"
    )
    print()
    print(
        f"  Samples with recall_required == 1.0: "
        f"{summary['perfect_required_samples']} / {n} "
        f"({100.0 * summary['perfect_required_ratio']:.1f}%)"
    )

    # Per-project breakdown
    by_project: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in per_sample:
        by_project[r["project"]].append(r)
    if len(by_project) > 1:
        print()
        print("  Per-project macro recall_required:")
        for proj, rows in sorted(by_project.items()):
            avg = sum(r["recall_required"] for r in rows) / max(len(rows), 1)
            print(f"    {proj:20s} {avg:.4f}   (n={len(rows)})")

    # Per-project: how many samples have at least one required class retrieved.
    # Samples without any required-labelled classes are excluded from the
    # denominator since the condition is undefined for them.
    print()
    print("  Per-project samples with >=1 required class retrieved:")
    print(
        f"    {'project':20s} {'hit':>5s} / {'evaluable':>9s}   {'ratio':>7s}   "
        f"({'no_req':>6s})"
    )
    by_proj_stats = summary.get("any_required_hit_by_project", {})
    for proj in sorted(by_proj_stats.keys()):
        s = by_proj_stats[proj]
        print(
            f"    {proj:20s} {s['samples_with_any_required_hit']:>5d} / "
            f"{s['samples_with_required']:>9d}   {s['ratio']:>7.4f}   "
            f"({s['samples_without_required']:>6d})"
        )
    overall = summary.get("any_required_hit_overall", {})
    if overall:
        print(
            f"    {'TOTAL':20s} {overall['samples_with_any_required_hit']:>5d} / "
            f"{overall['samples_with_required']:>9d}   {overall['ratio']:>7.4f}   "
            f"({overall['samples_without_required']:>6d})"
        )
    print("    (no_req = samples in the project that have 0 required-labelled classes)")


def _print_misses(results: list[dict[str, Any]], max_samples: int = 20) -> None:
    """Print samples where recall_required < 1.0, so we can see what's missed."""
    bad = [r for r in results if r["required_total"] > 0 and r["recall_required"] < 1.0]
    if not bad:
        print("\nNo samples with missed `required` classes ✓")
        return
    print(f"\n{len(bad)} sample(s) have missed required classes:")
    for r in bad[:max_samples]:
        print()
        print(
            f"  [{r['project']}] sample_id={r['sample_id']} "
            f"recall_required={r['recall_required']:.2f} "
            f"({r['required_found']}/{r['required_total']})"
        )
        print(f"    query: {r['query'][:140]}")
        for cls in r["missed_required"]:
            print(f"    MISS (required): {cls}")
        for cls in r["missed_useful"][:5]:
            print(f"    miss (useful):   {cls}")
    if len(bad) > max_samples:
        print(f"\n  ... and {len(bad) - max_samples} more samples with misses")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.verbose:
        # re-init logger with DEBUG
        import logging

        logging.getLogger("uml_pruner").setLevel(logging.DEBUG)

    cfg = load_config(args.config)
    emb_raw = cfg.get("embeddings") if hasattr(cfg, "get") else None

    model_name = (
        args.model
        or (emb_raw.get("model") if emb_raw else None)
        or "nomic-ai/nomic-embed-text-v1.5"
    )
    device = args.device or (emb_raw.get("device") if emb_raw else None) or "auto"
    batch_size = (
        args.batch_size or (emb_raw.get("batch_size") if emb_raw else None) or 32
    )
    cache_dir = Path(
        args.cache_dir
        or (emb_raw.get("cache_dir") if emb_raw else None)
        or "data/embeddings"
    )
    diagrams_dir = Path(
        args.diagrams_dir
        or (cfg.paths.get("diagrams_dir") if hasattr(cfg, "paths") else None)
        or "uml_with_methods"
    )

    default_top_k = (emb_raw.get("top_k") if emb_raw else None) or 300
    k_values: list[int] = args.top_k or [default_top_k]
    k_values = sorted(set(k_values))
    max_k = max(k_values)

    # Load samples (dataset is treated as immutable: no merging or filtering here)
    samples = load_dataset(args.dataset)
    if args.repo:
        samples = [s for s in samples if s.repo == args.repo]
    if args.sample_id:
        samples = [s for s in samples if s.sample_id == args.sample_id]
    if args.limit:
        samples = samples[: args.limit]

    if not samples:
        logger.error("No samples to evaluate.")
        sys.exit(1)

    logger.info(
        "Evaluating %d sample(s) at top_k=%s using model '%s' (device=%s)",
        len(samples),
        k_values,
        model_name,
        device,
    )

    # Init encoder once (lazy internally)
    encoder = LocalEncoder(
        EncoderConfig(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )
    )

    # Cache indices per-project (file stem) so we load each at most once
    index_cache: dict[str, Any] = {}
    diagram_cache: dict[str, dict] = {}

    # Per-k accumulators
    per_k_results: dict[int, list[dict[str, Any]]] = {k: [] for k in k_values}
    skipped: list[dict[str, Any]] = []

    for sample in samples:
        fname = diagram_filename_for_repo(sample.repo)
        if not fname:
            skipped.append(
                {
                    "sample_id": sample.sample_id,
                    "reason": "unknown_repo",
                    "repo": sample.repo,
                    "central_node": sample.central_node,
                }
            )
            continue
        diagram_name = Path(fname).stem

        # Load diagram (only for hash validation)
        if fname not in diagram_cache:
            path = diagrams_dir / fname
            if not path.exists():
                skipped.append(
                    {
                        "sample_id": sample.sample_id,
                        "reason": "diagram_missing",
                        "path": str(path),
                    }
                )
                continue
            diagram_cache[fname] = load_diagram(path)
        diagram = diagram_cache[fname]

        # Load index (once per diagram)
        if diagram_name not in index_cache:
            entry = load_cache(cache_dir, diagram_name)
            if entry is None:
                skipped.append(
                    {
                        "sample_id": sample.sample_id,
                        "reason": "index_missing",
                        "diagram": diagram_name,
                    }
                )
                index_cache[diagram_name] = None
                continue
            expected_hash = compute_diagram_hash(diagram["nodes"])
            if not is_valid(
                entry, expected_model=model_name, expected_diagram_hash=expected_hash
            ):
                skipped.append(
                    {
                        "sample_id": sample.sample_id,
                        "reason": "index_stale",
                        "diagram": diagram_name,
                    }
                )
                index_cache[diagram_name] = None
                continue
            index_cache[diagram_name] = entry

        entry = index_cache[diagram_name]
        if entry is None:
            # previous attempt failed
            skipped.append(
                {
                    "sample_id": sample.sample_id,
                    "reason": "index_unavailable",
                    "diagram": diagram_name,
                }
            )
            continue

        # Single retrieval at max_k; derive smaller K by slicing the ranked list.
        hits = retrieve_top_k(sample.query, entry, encoder, top_k=max_k)
        ranked_ids = [h.node_id for h in hits]

        for k in k_values:
            ids_at_k = set(ranked_ids[:k])
            per_k_results[k].append(_eval_sample(sample, ids_at_k, k))

    # Report each K
    report: dict[str, Any] = {
        "config": {
            "model": model_name,
            "device": device,
            "cache_dir": str(cache_dir),
            "k_values": k_values,
            "num_samples_input": len(samples),
            "num_samples_skipped": len(skipped),
        },
        "skipped": skipped,
        "per_k": {},
    }

    for k in k_values:
        results = per_k_results[k]
        summary = _aggregate(results)
        report["per_k"][str(k)] = {
            "summary": summary,
            "per_sample": results,
        }
        _print_summary(k, summary, results)
        if args.show_misses:
            _print_misses(results)

    if skipped:
        print()
        print(f"Skipped {len(skipped)} sample(s):")
        reasons: dict[str, int] = defaultdict(int)
        for s in skipped:
            reasons[s["reason"]] += 1
        for r, n in reasons.items():
            print(f"  {r}: {n}")

    if args.output:
        save_json(report, args.output)
        print(f"\nFull JSON report written to {args.output}")


if __name__ == "__main__":
    main()
