"""Approach #3: agentic chunking + per-chunk relevance selection.

High-level steps:

    1. **Chunk**. Split the full diagram into smaller "files" — sets of
       classes that share a package / are structurally close. Each chunk is
       small enough to fit in the LLM context with room to think.
    2. **Survey**. Run an LLM agent over chunks (in parallel where possible).
       For each chunk it answers: "are any of these classes relevant to the
       query? if yes, which ones and at what confidence?".
    3. **Synthesize**. Collect every chunk's positive answers into one
       candidate set, then ask the agent to deduplicate and finalize the
       REQUIRED / USEFUL split.

Why it might beat #1 and #2:
    - Always sees the *whole* diagram (chunked, but no information dropped at
      the retrieval stage), so it can't miss a relevant class because the
      embedding retriever didn't surface it.
    - Naturally parallelizable across chunks.

Cost trade-off: O(diagram_size) LLM calls. Suitable when accuracy matters
more than budget.

This file is currently a stub. Plan and TODOs live in
``scripts/approaches/agentic_chunks/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.types import ApproachInputs, ApproachResult

NAME = "agentic_chunks"


@dataclass
class AgenticChunksConfig:
    chunk_strategy: str = "package"  # "package" | "size" | "louvain"
    chunk_size: int = 80  # target classes per chunk for "size" strategy
    max_parallel_chunks: int = 4
    survey_max_output_per_chunk: int = 30
    # If True, run a final synthesizer LLM pass over the combined candidates.
    run_synthesizer: bool = True


class AgenticChunksRunner:
    name = NAME

    def __init__(self, cfg: AgenticChunksConfig) -> None:
        self._cfg = cfg

    async def run(self, inputs: ApproachInputs) -> ApproachResult:  # pragma: no cover
        raise NotImplementedError(
            "approach 'agentic_chunks' is not implemented yet — see "
            "scripts/approaches/agentic_chunks/README.md for the plan."
        )

    async def aclose(self) -> None:
        return None


def build_runner(cfg: Any | None = None) -> AgenticChunksRunner:
    return AgenticChunksRunner(AgenticChunksConfig())
