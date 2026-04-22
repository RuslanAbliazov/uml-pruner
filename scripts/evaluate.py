#!/usr/bin/env python3
"""CLI: evaluate pipeline results against annotations.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import evaluate_test_set, format_summary_report
from src.utils.io import save_json
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pruning results.")
    parser.add_argument(
        "--annotations",
        default="annotations.csv",
        help="Path to annotations CSV (default: annotations.csv).",
    )
    parser.add_argument(
        "--results-dir",
        default="data/results",
        help="Directory with pipeline result JSON files named '{sample_id}.json'.",
    )
    parser.add_argument(
        "--filename-template",
        default="{sample_id}.json",
        help="Filename template for results (default '{sample_id}.json').",
    )
    parser.add_argument(
        "--output",
        default="data/results/evaluation_report.json",
        help="Where to write the full JSON report.",
    )
    parser.add_argument(
        "--include-non-finalized",
        action="store_true",
        help="Include samples whose status is not 'Finalized'.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(level="DEBUG" if args.verbose else "INFO")

    result = evaluate_test_set(
        annotations_csv=args.annotations,
        results_dir=args.results_dir,
        result_filename_template=args.filename_template,
        finalized_only=not args.include_non_finalized,
    )

    report_text = format_summary_report(result)
    print(report_text)

    save_json(result.to_dict(), args.output)
    print(f"\nFull JSON report written to {args.output}")


if __name__ == "__main__":
    main()
