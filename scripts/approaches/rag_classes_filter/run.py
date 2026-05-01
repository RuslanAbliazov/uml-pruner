#!/usr/bin/env python3
"""Thin wrapper that runs ``scripts/benchmark.py`` with this approach selected.

For all available knobs (filtering, evaluation, etc.) see
``python scripts/benchmark.py --help``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = PROJECT_ROOT / "scripts" / "benchmark.py"


def main() -> None:
    args = sys.argv[1:]
    # Inject --approach if the user didn't already specify it.
    if "--approach" not in args:
        args = ["--approach", "rag_classes_filter", *args]
    os.execv(sys.executable, [sys.executable, str(BENCHMARK), *args])


if __name__ == "__main__":
    main()
