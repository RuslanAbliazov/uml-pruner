#!/usr/bin/env python3
"""Compute inter-annotator agreement on raw ``annotations.csv``.

Important: this works on the RAW annotations file, NOT the consolidated
``data/dataset.csv``. The consolidated file has already merged votes and
dropped ``irrelevant`` labels — the multi-annotator signal IAA needs is
gone there.

Usage:
    # Defaults: annotations.csv at the repo root, finalized rows only.
    python scripts/iaa.py

    # Custom path, also include in-progress rows:
    python scripts/iaa.py --annotations annotations.csv --include-non-finalized

    # Drop specific annotators (mirror of build_dataset.py).
    python scripts/iaa.py --exclude-annotator AndrewRatkov

    # Use intersection of labelled nodes instead of union+implicit-irrelevant:
    python scripts/iaa.py --policy intersection

    # Save full per-sample report:
    python scripts/iaa.py --output data/results/iaa.json

What you get
------------

Stdout: a short summary (mean κ, mean agreement, sample count).
With ``--output``: a JSON file with per-sample pairwise kappas plus the
summary.

Reading the numbers (Landis & Koch 1977 conventions):
    κ ≤ 0.20  poor / slight
    0.21–0.40 fair
    0.41–0.60 moderate
    0.61–0.80 substantial
    0.81–1.00 almost perfect

If the mean κ is around 0.5, your evaluation ceiling is around there too —
no algorithm can outperform the average annotator-to-annotator agreement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.iaa import (  # noqa: E402
    compute_iaa,
    format_summary,
    load_raw_annotations,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inter-annotator agreement on raw annotations.csv."
    )
    p.add_argument(
        "--annotations",
        default="annotations.csv",
        help="Path to raw annotations CSV (default: annotations.csv).",
    )
    p.add_argument(
        "--include-non-finalized",
        action="store_true",
        help='Also include rows whose status != "Finalized".',
    )
    p.add_argument(
        "--exclude-annotator",
        action="append",
        default=[],
        metavar="NAME",
        help="Drop votes from this annotator. May be passed multiple times.",
    )
    p.add_argument(
        "--policy",
        choices=("union_with_implicit_irrelevant", "intersection"),
        default="union_with_implicit_irrelevant",
        help="How to align label arrays for two annotators (default: union).",
    )
    p.add_argument(
        "--min-annotators",
        type=int,
        default=2,
        help="Skip samples with fewer than this many annotators (default: 2).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="If set, write the full report (per-sample + summary) as JSON.",
    )
    p.add_argument(
        "--binary",
        action="store_true",
        default=False,
        help="Merge required and useful into a single positive class "
        "(required_or_useful) and treat vs. irrelevant.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_raw_annotations(
        args.annotations,
        finalized_only=not args.include_non_finalized,
        exclude_annotators=args.exclude_annotator,
    )
    if not rows:
        print(
            "[error] no usable annotation rows found in "
            f"{args.annotations}. "
            "Tip: check --include-non-finalized and --exclude-annotator.",
            file=sys.stderr,
        )
        sys.exit(2)

    report = compute_iaa(
        rows,
        policy=args.policy,
        min_annotators=args.min_annotators,
        merge_required_useful=args.binary,
    )
    print(format_summary(report))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nFull report: {out}")


if __name__ == "__main__":
    main()