"""Approach #2: anchor selection + neighborhood expansion + pruning.

The pipeline is split across small, single-purpose modules:

    config.py         AnchorNeighborsConfig + ``build_runner`` factory.
    candidates.py     Stage 1 — RAG retrieval.
    select_anchor.py  Stage 2 — LLM picks one anchor.
    expand.py         Stage 3 — collect 1-hop neighborhood.
    prune.py          Stage 4 — LLM REQUIRED/USEFUL/IRRELEVANT.
    runner.py         Glue that wires the four stages together.
    prompts.py        Tiny wrappers over ./prompts/*.txt.

For local execution use ``run.py`` in this directory.
"""

from src.approaches.anchor_neighbors.config import build_runner
from src.approaches.anchor_neighbors.runner import NAME

__all__ = ["NAME", "build_runner"]
