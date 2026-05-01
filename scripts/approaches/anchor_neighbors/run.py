#!/usr/bin/env python3
"""Thin wrapper around ``scripts/benchmark.py`` for approach #2."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = PROJECT_ROOT / "scripts" / "benchmark.py"


def main() -> None:
    args = sys.argv[1:]
    if "--approach" not in args:
        args = ["--approach", "anchor_neighbors", *args]
    os.execv(sys.executable, [sys.executable, str(BENCHMARK), *args])


if __name__ == "__main__":
    main()
