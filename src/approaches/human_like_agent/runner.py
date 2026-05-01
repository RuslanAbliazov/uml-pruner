"""Approach #4: human-like agent.

Mimics how a human investigator would explore an unfamiliar codebase:

    1. **Identify anchors**. The agent reads the query and asks: "which N
       classes are most likely the entry points / central pieces?" — chosen
       via embedding retrieval and (optionally) a small LLM rerank.
    2. **Expand by centrality**. Around each anchor, surface neighbors with
       high *betweenness centrality* and/or many *call_in_code* hits. These
       are the structurally important classes a human would naturally read
       next.
    3. **Prune**. The agent inspects the expanded neighborhood and drops
       classes that look incidental, keeping the rest as REQUIRED / USEFUL.

Why it might beat the other approaches:
    - Mirrors how good annotators actually work — the dataset itself was
      built using betweenness/calls thresholds (see annotations.csv
      ``slider_state``), so leaning on the same signals should align well
      with the ground truth.
    - Composable: anchors come from RAG, expansion from the graph, pruning
      from the LLM. Each piece can be ablated.

This file is currently a stub. Plan and TODOs live in
``scripts/approaches/human_like_agent/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.approaches.base import ApproachInputs, ApproachResult

NAME = "human_like_agent"


@dataclass
class HumanLikeAgentConfig:
    n_anchors: int = 5
    # Centrality knobs (mirroring the slider used by human annotators)
    bc_threshold: float = 0.0005
    calls_threshold: int = 1
    # Cap on neighbors per anchor expanded by centrality
    max_neighbors_per_anchor: int = 30
    # If True, do a final LLM prune pass on the expanded neighborhood
    run_prune_llm: bool = True


class HumanLikeAgentRunner:
    name = NAME

    def __init__(self, cfg: HumanLikeAgentConfig) -> None:
        self._cfg = cfg

    async def run(self, inputs: ApproachInputs) -> ApproachResult:  # pragma: no cover
        raise NotImplementedError(
            "approach 'human_like_agent' is not implemented yet — see "
            "scripts/approaches/human_like_agent/README.md for the plan."
        )

    async def aclose(self) -> None:
        return None


def build_runner(cfg: Any | None = None) -> HumanLikeAgentRunner:
    return HumanLikeAgentRunner(HumanLikeAgentConfig())
