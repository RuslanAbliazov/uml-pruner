"""Anchor + neighbors + prune — orchestrator.

Four stages, each implemented in its own module:

    candidates.py     -> RAG: top-K candidates for the query
    select_anchor.py  -> LLM picks ONE candidate as the anchor
    expand.py         -> collect 1-hop neighborhood (with optional cap)
    prune.py          -> LLM tags the subgraph as REQUIRED / USEFUL / ...

The runner is just glue: it owns the resources (LLM client, candidate
finder) and stitches the stages together into the standard
``ApproachResult``.
"""

from __future__ import annotations

from src.approaches._common.compressor import filter_subgraph
from src.approaches.anchor_neighbors.candidates import CandidateFinder
from src.approaches.anchor_neighbors.config import AnchorNeighborsConfig
from src.approaches.anchor_neighbors.expand import build_subgraph
from src.approaches.anchor_neighbors.prune import prune
from src.approaches.anchor_neighbors.select_anchor import select_anchor
from src.core.types import ApproachInputs, ApproachResult
from src.llm.client import LLMClient

NAME = "anchor_neighbors"


class AnchorNeighborsRunner:
    """Implements approach #2 against the unified ``ApproachRunner`` interface."""

    name = NAME

    def __init__(self, cfg: AnchorNeighborsConfig, llm_client: LLMClient) -> None:
        self._cfg = cfg
        self._llm = llm_client
        self._finder = CandidateFinder(
            model=cfg.embedding_model,
            device=cfg.embedding_device,
            batch_size=cfg.embedding_batch_size,
            cache_dir=cfg.embedding_cache_dir,
        )

    async def run(self, inputs: ApproachInputs) -> ApproachResult:
        nodes = inputs.nodes
        edges = inputs.edges
        node_by_id = {n["node_id"]: n for n in nodes if n.get("node_id")}
        diagram_stem = _diagram_stem(inputs)

        # 1. Candidates.
        candidates = self._finder.fetch(
            query=inputs.query,
            diagram_stem=diagram_stem,
            nodes=nodes,
            top_k=self._cfg.n_candidates,
        )
        if not candidates:
            return _empty(reason="no_candidates",
                          metadata={"diagram_stem": diagram_stem})

        # 2. Anchor.
        anchor_id, anchor_reason = await select_anchor(
            query=inputs.query,
            candidates=candidates,
            node_by_id=node_by_id,
            llm=self._llm,
        )
        if not anchor_id:
            return _empty(
                reason="anchor_selection_failed",
                metadata={
                    "diagram_stem": diagram_stem,
                    "candidates": [c["node_id"] for c in candidates],
                },
            )

        # 3. Expand.
        sub_nodes, sub_edges, expand_info = build_subgraph(
            anchor=anchor_id,
            nodes=nodes,
            edges=edges,
            node_by_id=node_by_id,
            cap=self._cfg.max_subgraph_nodes,
        )

        # 4. Prune.
        required_ids, useful_ids = await prune(
            query=inputs.query,
            anchor=anchor_id,
            nodes=sub_nodes,
            edges=sub_edges,
            llm=self._llm,
        )
        # Anchor is implicitly relevant — guarantee it survives.
        if anchor_id not in required_ids and anchor_id not in useful_ids:
            required_ids.add(anchor_id)

        kept_ids = required_ids | useful_ids
        final_nodes, final_edges = filter_subgraph(nodes, edges, kept_ids)

        return ApproachResult(
            approach=self.name,
            nodes=final_nodes,
            edges=final_edges,
            required_node_ids=sorted(required_ids),
            useful_node_ids=sorted(useful_ids),
            metadata={
                "diagram_stem": diagram_stem,
                "n_candidates_requested": self._cfg.n_candidates,
                "n_candidates_returned": len(candidates),
                "candidates": [
                    {"node_id": c["node_id"], "score": c.get("score")}
                    for c in candidates
                ],
                "anchor": anchor_id,
                "anchor_reason": anchor_reason,
                "neighbors_count": expand_info["neighbors_count"],
                "subgraph_node_count": len(sub_nodes),
                "subgraph_edge_count": len(sub_edges),
                "subgraph_truncated": expand_info["subgraph_truncated"],
                "filtered_node_count": len(final_nodes),
            },
        )

    async def aclose(self) -> None:
        # No long-lived resources; encoder is in-process only.
        return None


# ---------------------------------------------------------------------------
# Small helpers (private)
# ---------------------------------------------------------------------------


def _diagram_stem(inputs: ApproachInputs) -> str:
    """Repo basename used to locate ``data/embeddings/<stem>``."""
    if inputs.repo and "/" in inputs.repo:
        return inputs.repo.split("/", 1)[1]
    return inputs.repo or ""


def _empty(*, reason: str, metadata: dict) -> ApproachResult:
    meta = dict(metadata)
    meta["aborted"] = reason
    return ApproachResult(
        approach=NAME,
        nodes=[],
        edges=[],
        required_node_ids=[],
        useful_node_ids=[],
        metadata=meta,
    )
