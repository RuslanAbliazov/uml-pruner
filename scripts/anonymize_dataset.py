#!/usr/bin/env python3
"""Strip ground-truth columns from a dataset CSV.

Pipelines must only ever see ``repo`` and ``query`` — the same information a
real user would supply. Keeping ``central_node``, ``entity_annotations`` (or
any other oracle field) inside the file we feed into a runner risks subtle
data leakage: a careless ``DictReader`` row-pass could let those keys reach
the LLM payload, the prompt template, or the logs.

This script reads a full annotated dataset (typically ``data/dataset.csv``,
produced by ``scripts/build_dataset.py``) and writes a slim copy that
contains only ``repo`` and ``query``. Row order is preserved so the slim
file aligns 1:1 with the original on ``sample_id`` if you keep both around.

Examples
--------
Default in/out paths (data/dataset.csv -> data/dataset_queries.csv):

    python scripts/anonymize_dataset.py

Custom paths:

    python scripts/anonymize_dataset.py \\
        --input data/dataset_all.csv \\
        --output data/dataset_all_queries.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

KEEP_COLUMNS = ("repo", "query")


def anonymize(input_path: Path, output_path: Path) -> int:
    """Copy ``input_path`` to ``output_path`` keeping only ``repo,query``.

    Returns the number of data rows written.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} has no header row")

        missing = [c for c in KEEP_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"{input_path} is missing required column(s): {missing}. "
                f"Found: {reader.fieldnames}"
            )

        with output_path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=list(KEEP_COLUMNS))
            writer.writeheader()

            written = 0
            for row in reader:
                writer.writerow({col: row[col] for col in KEEP_COLUMNS})
                written += 1

    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Produce a leakage-safe copy of the dataset containing only "
            "the columns a real user would provide (repo, query)."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/dataset.csv"),
        help="Path to the full annotated dataset CSV (default: data/dataset.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dataset_queries.csv"),
        help="Where to write the slim dataset (default: data/dataset_queries.csv)",
    )
    args = parser.parse_args()

    n = anonymize(args.input, args.output)
    print(
        f"Wrote {n} row(s) to {args.output} "
        f"(columns: {', '.join(KEEP_COLUMNS)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
