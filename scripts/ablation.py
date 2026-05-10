#!/usr/bin/env python3
"""Run several approaches over the dataset and print a side-by-side table.

The table compares F1, recall (required + useful + overall), precision,
and mean output size across approaches. Pairs naturally with the bootstrap
confidence-interval column (if requested) so you can see whether a 0.02
F1 difference is real on your dataset.

Usage:
    # Compare query-agnostic baselines and the real approach:
    python scripts/ablation.py \\
        --approaches empty full_diagram random_subset top_degree \\
                     anchor_neighbors

    # Add bootstrap 95% CIs (slower, but the right thing to report):
    python scripts/ablation.py \\
        --approaches empty random_subset anchor_neighbors \\
        --bootstrap 1000

    # Restrict to a subset of samples, save the full table as JSON:
    python scripts/ablation.py \\
        --approaches anchor_neighbors empty \\
        --limit 10 \\
        --output data/results/ablation.json

The script does NOT regenerate result files that already exist (so reruns
are cheap). Pass ``--overwrite`` to force regeneration.

Note on oracles: oracle baselines (``oracle_central_plus_neighbors``,
``oracle_gold_only``) are NOT registered approaches — run them via
``scripts/run_oracle_baselines.py`` first, then point this script at their
output directories with ``--include-existing-dir``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.approaches import get_runner, list_approaches  # noqa: E402
from src.core.config import load_config  # noqa: E402
from src.core.io import load_diagram, save_diagram, save_json  # noqa: E402
from src.core.types import ApproachInputs  # noqa: E402
from src.eval.annotations import diagram_filename_for_repo, load_dataset  # noqa: E402
from src.eval.evaluator import EvaluationResult, evaluate_test_set  # noqa: E402
from src.eval.metrics import SampleMetrics  # noqa: E402


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------


async def _generate_for_approach(
    approach_name: str,
    samples: list[Any],
    diagrams_dir: Path,
    output_dir: Path,
    cfg: Any,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = get_runner(approach_name, cfg)

    diagram_cache: dict[str, dict] = {}
    for sample in samples:
        out_file = output_dir / f"{sample.sample_id}.json"
        if out_file.exists() and not overwrite:
            continue

        fname = diagram_filename_for_repo(sample.repo)
        path = diagrams_dir / fname
        if not path.exists():
            continue
        if fname not in diagram_cache:
            diagram_cache[fname] = load_diagram(path)

        inputs = ApproachInputs(
            query=sample.query,
            diagram=diagram_cache[fname],
            sample_id=sample.sample_id,
            repo=sample.repo,
        )
        try:
            result = await runner.run(inputs)
        except NotImplementedError:
            print(f"  [skip] approach '{approach_name}' is a stub")
            await runner.aclose()
            return
        except Exception as e:  # noqa: BLE001 — surface failures, don't crash the table
            print(f"  [warn] {approach_name}/{sample.sample_id}: {e!r}")
            continue

        diagram_out = result.to_diagram()
        meta = diagram_out.setdefault("metadata", {})
        meta.update(
            {
                "approach": approach_name,
                "sample_id": sample.sample_id,
                "repo": sample.repo,
                "query": sample.query,
            }
        )
        save_diagram(diagram_out, out_file)

    await runner.aclose()


# ----------------------------------------------------------------------------
# Bootstrap CI
# ----------------------------------------------------------------------------


def _bootstrap_ci(
    values: list[float],
    n_iter: int,
    seed: int = 12345,
    pcts: tuple[float, float] = (0.025, 0.975),
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    With ~28 samples, 1000–5000 iterations is enough; the bottleneck is
    the underlying generation, not this loop.
    """
    if not values or n_iter <= 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_iter):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(pcts[0] * n_iter)]
    hi = means[min(int(pcts[1] * n_iter), n_iter - 1)]
    return (lo, hi)


# ----------------------------------------------------------------------------
# Table
# ----------------------------------------------------------------------------


COLUMNS = (
    "approach",
    "n_eval",
    "f1_mean",
    "f1_ci",
    "recall_required",
    "recall_useful",
    "recall_overall",
    "precision_known",
    "precision_strict",
    "mean_size",
)


def _per_approach_row(
    name: str,
    result: EvaluationResult,
    bootstrap: int,
) -> dict[str, Any]:
    if not result.per_sample:
        return {"approach": name, "n_eval": 0}

    f1s = [m.f1_score for m in result.per_sample]
    sizes = [m.predicted_count for m in result.per_sample]

    macro = result.summary["macro"]
    row = {
        "approach": name,
        "n_eval": result.summary["num_samples"],
        "f1_mean": round(macro["mean_f1_score"], 4),
        "recall_required": round(macro["mean_recall_required"], 4),
        "recall_useful": round(macro["mean_recall_useful"], 4),
        "recall_overall": round(macro["mean_recall_overall"], 4),
        "precision_known": round(macro["mean_precision_known"], 4),
        "precision_strict": round(macro["mean_precision_strict"], 4),
        "mean_size": round(macro["mean_predicted_size"], 2),
    }
    if bootstrap > 0:
        lo, hi = _bootstrap_ci(f1s, bootstrap)
        row["f1_ci"] = f"[{lo:.3f}, {hi:.3f}]"
    else:
        row["f1_ci"] = "—"
    row["_per_sample_f1"] = f1s
    row["_per_sample_sizes"] = sizes
    return row


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Pretty-print a fixed-column table to stdout."""
    if not rows:
        print("(no rows)")
        return

    widths = {
        "approach": max(20, max(len(r.get("approach", "")) for r in rows)),
        "n_eval": 6,
        "f1_mean": 8,
        "f1_ci": 18,
        "recall_required": 9,
        "recall_useful": 9,
        "recall_overall": 9,
        "precision_known": 9,
        "precision_strict": 9,
        "mean_size": 9,
    }
    header_map = {
        "approach": "approach",
        "n_eval": "n",
        "f1_mean": "F1",
        "f1_ci": "F1 CI95%",
        "recall_required": "rec_req",
        "recall_useful": "rec_use",
        "recall_overall": "rec_all",
        "precision_known": "p_known",
        "precision_strict": "p_strict",
        "mean_size": "size",
    }
    line = " | ".join(
        f"{header_map[c]:<{widths[c]}}" if c == "approach"
        else f"{header_map[c]:>{widths[c]}}"
        for c in COLUMNS
    )
    print("=" * len(line))
    print(line)
    print("-" * len(line))
    for r in rows:
        cells = []
        for c in COLUMNS:
            v = r.get(c, "")
            if c == "approach":
                cells.append(f"{str(v):<{widths[c]}}")
            else:
                cells.append(f"{str(v):>{widths[c]}}")
        print(" | ".join(cells))
    print("=" * len(line))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Side-by-side ablation across approaches.")
    p.add_argument(
        "--approaches",
        nargs="+",
        required=True,
        help="Approach names registered in src.approaches.REGISTRY.",
    )
    p.add_argument("--dataset", default="data/dataset.csv")
    p.add_argument("--diagrams-dir", default="data/diagrams_normalized")
    p.add_argument(
        "--output-root",
        default="data/results",
        help="Per-approach generation outputs go to <root>/<approach>/.",
    )
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--repo", default="")
    p.add_argument("--sample-id", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Bootstrap iterations for F1 95%% CI (0 = disable).",
    )
    p.add_argument(
        "--include-existing-dir",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Add a results directory that was generated outside this script "
            "(e.g. oracle baselines). Format: NAME=PATH. May be passed "
            "multiple times."
        ),
    )
    p.add_argument(
        "--output",
        default=None,
        help="Save the full table as JSON.",
    )
    p.add_argument(
        "--skip-generation",
        action="store_true",
        help="Don't run any approach; only score existing result dirs.",
    )
    return p.parse_args()


async def _run(args: argparse.Namespace) -> int:
    # Validate names
    registered = set(list_approaches())
    bad = [a for a in args.approaches if a not in registered]
    if bad:
        print(
            f"[error] unknown approaches: {bad}. "
            f"Registered: {sorted(registered)}",
            file=sys.stderr,
        )
        return 2

    # Slice the dataset
    samples = load_dataset(args.dataset)
    if args.repo:
        samples = [s for s in samples if s.repo == args.repo]
    if args.sample_id:
        samples = [s for s in samples if s.sample_id == args.sample_id]
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        print("[error] no samples after filtering.", file=sys.stderr)
        return 2

    # Generate results for each requested approach
    diagrams_dir = Path(args.diagrams_dir)
    output_root = Path(args.output_root)
    cfg = None
    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg = load_config(args.config)

    if not args.skip_generation:
        for name in args.approaches:
            out_dir = output_root / name
            print(f"\n=== Generating for approach '{name}' -> {out_dir} ===")
            t0 = time.time()
            await _generate_for_approach(
                approach_name=name,
                samples=samples,
                diagrams_dir=diagrams_dir,
                output_dir=out_dir,
                cfg=cfg,
                overwrite=args.overwrite,
            )
            print(f"  done in {time.time() - t0:.1f}s")

    # Score everything
    rows: list[dict[str, Any]] = []
    eval_dirs = [(name, output_root / name) for name in args.approaches]
    for entry in args.include_existing_dir:
        if "=" not in entry:
            print(f"[warn] ignoring malformed --include-existing-dir: {entry!r}")
            continue
        nm, pth = entry.split("=", 1)
        eval_dirs.append((nm, Path(pth)))

    for name, dir_ in eval_dirs:
        if not dir_.exists():
            print(f"[warn] result dir not found for '{name}': {dir_}")
            rows.append({"approach": name, "n_eval": 0})
            continue
        print(f"\n=== Evaluating '{name}' from {dir_} ===")
        result = evaluate_test_set(args.dataset, str(dir_))
        rows.append(_per_approach_row(name, result, args.bootstrap))

    print()
    _print_table(rows)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Drop verbose internals from the saved JSON.
        for r in rows:
            r.pop("_per_sample_f1", None)
            r.pop("_per_sample_sizes", None)
        out.write_text(
            json.dumps(
                {
                    "rows": rows,
                    "n_samples": len(samples),
                    "bootstrap": args.bootstrap,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved: {out}")

    return 0


def main() -> None:
    args = parse_args()
    rc = asyncio.run(_run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
