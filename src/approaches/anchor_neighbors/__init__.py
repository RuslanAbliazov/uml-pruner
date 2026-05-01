"""Approach #2: anchor selection + neighborhood expansion + pruning.

Implementation lives in :mod:`runner`. See ``README.md`` in
``scripts/approaches/anchor_neighbors/`` for the user-facing description.
"""

from src.approaches.anchor_neighbors.runner import NAME, build_runner

__all__ = ["NAME", "build_runner"]
