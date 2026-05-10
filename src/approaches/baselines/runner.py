"""Query-agnostic baseline runners.

Every runner here implements the standard :class:`ApproachRunner` protocol
(``async def run(inputs) -> ApproachResult``) and produces output in the
same shape as the real pipelines, so the standard evaluator and
``scripts/run.py`` work without changes.

Important: none of these baselines look at the query. They exist purely as
floor/ceiling references for the "real" approaches:

    * ``empty``         — F1 floor when predicting nothing.
    * ``full_diagram``  — F1 floor when predicting everything (recall=1.0).
    * ``random_subset`` — what F1 would you get if you didn't try at all.
    * ``top_degree``    — what F1 would you get if you used pure structure.

Why include these in a serious benchmark: they make claims like "our
approach reaches F1 = 0.40" interpretable. If ``random_subset`` of the same
size already reaches F1 = 0.35, the "real" approach is contributing very
little. If ``random_subset`` reaches F1 = 0.05, the contribution is real.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from typing import Any

from src.core.types import ApproachInputs, ApproachResult


# ---- helpers ---------------------------------------------------------------


def _all_node_ids(inputs: ApproachInputs) -> list[str]:
    """Return all valid node_ids in a deterministic order (insertion-order)."""
    out: list[str] = []
    seen: set[str] = set()
    for n in inputs.nodes:
        nid = n.get("node_id")
        if isinstance(nid, str) and nid and nid not in seen:
            out.append(nid)
            seen.add(nid)
    return out


def _select_subgraph(
    inputs: ApproachInputs, keep: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter nodes/edges down to ``keep``. Edges retained iff both ends kept."""
    nodes_out = [n for n in inputs.nodes if n.get("node_id") in keep]
    edges_out = [
        e
        for e in inputs.edges
        if e.get("node_id_from") in keep and e.get("node_id_to") in keep
    ]
    return nodes_out, edges_out


def _stable_seed(base_seed: int, sample_id: str) -> int:
    """Combine a base seed with a sample_id into a stable per-sample seed.

    We hash the sample_id so that rerunning the same sample is bit-exact
    even across machines, and so that the result doesn't depend on Python's
    hash randomization (PYTHONHASHSEED).
    """
    h = hashlib.blake2b(sample_id.encode("utf-8"), digest_size=8).hexdigest()
    return base_seed ^ int(h, 16)


# ---- runners ---------------------------------------------------------------


class EmptyBaseline:
    """Predicts the empty subgraph. Recall = 0, precision undefined (= 0)."""

    name = "empty"

    async def run(self, inputs: ApproachInputs) -> ApproachResult:
        return ApproachResult(
            approach=self.name,
            nodes=[],
            edges=[],
            required_node_ids=[],
            useful_node_ids=[],
            metadata={"strategy": "empty"},
        )

    async def aclose(self) -> None:
        return None


class FullDiagramBaseline:
    """Predicts the entire diagram. Recall = 1.0; precision tiny.

    The output ``required_node_ids`` is the full node set; ``useful_node_ids``
    is empty (the split is meaningless here, but we have to pick one).
    """

    name = "full_diagram"

    async def run(self, inputs: ApproachInputs) -> ApproachResult:
        all_ids = _all_node_ids(inputs)
        return ApproachResult(
            approach=self.name,
            nodes=list(inputs.nodes),
            edges=list(inputs.edges),
            required_node_ids=sorted(all_ids),
            useful_node_ids=[],
            metadata={
                "strategy": "full_diagram",
                "n_nodes": len(all_ids),
                "n_edges": len(inputs.edges),
            },
        )

    async def aclose(self) -> None:
        return None


class RandomSubsetBaseline:
    """Predicts ``size`` nodes chosen uniformly at random per sample.

    Seeded by ``(base_seed, sample_id)`` for reproducibility. Two runs with
    the same dataset produce the same predictions.
    """

    name = "random_subset"

    def __init__(self, size: int = 5, seed: int = 42) -> None:
        if size < 0:
            raise ValueError(f"size must be >= 0, got {size}")
        self._size = size
        self._seed = seed

    async def run(self, inputs: ApproachInputs) -> ApproachResult:
        ids = _all_node_ids(inputs)
        if not ids or self._size == 0:
            return ApproachResult(
                approach=self.name,
                metadata={
                    "strategy": "random_subset",
                    "size": self._size,
                    "n_available": len(ids),
                },
            )
        k = min(self._size, len(ids))
        rng = random.Random(_stable_seed(self._seed, inputs.sample_id))
        chosen = set(rng.sample(ids, k=k))
        nodes_out, edges_out = _select_subgraph(inputs, chosen)
        return ApproachResult(
            approach=self.name,
            nodes=nodes_out,
            edges=edges_out,
            required_node_ids=sorted(chosen),
            useful_node_ids=[],
            metadata={
                "strategy": "random_subset",
                "size": self._size,
                "n_available": len(ids),
                "n_picked": k,
                "seed_base": self._seed,
            },
        )

    async def aclose(self) -> None:
        return None


class TopDegreeBaseline:
    """Predicts the top-``size`` nodes by total degree (in + out edges).

    Self-loops contribute 0 (they tell us nothing about external structure).
    Tie-break: lexicographic on node_id so output is deterministic.
    """

    name = "top_degree"

    def __init__(self, size: int = 5) -> None:
        if size < 0:
            raise ValueError(f"size must be >= 0, got {size}")
        self._size = size

    async def run(self, inputs: ApproachInputs) -> ApproachResult:
        ids = _all_node_ids(inputs)
        if not ids or self._size == 0:
            return ApproachResult(
                approach=self.name,
                metadata={
                    "strategy": "top_degree",
                    "size": self._size,
                    "n_available": len(ids),
                },
            )

        deg: Counter[str] = Counter()
        valid_ids = set(ids)
        for e in inputs.edges:
            a = e.get("node_id_from")
            b = e.get("node_id_to")
            if not a or not b or a == b:
                continue
            if a in valid_ids:
                deg[a] += 1
            if b in valid_ids:
                deg[b] += 1
        # ranked: degree DESC, then node_id ASC
        ranked = sorted(ids, key=lambda nid: (-deg.get(nid, 0), nid))
        chosen = set(ranked[: self._size])

        nodes_out, edges_out = _select_subgraph(inputs, chosen)
        return ApproachResult(
            approach=self.name,
            nodes=nodes_out,
            edges=edges_out,
            required_node_ids=sorted(chosen),
            useful_node_ids=[],
            metadata={
                "strategy": "top_degree",
                "size": self._size,
                "n_available": len(ids),
                "n_picked": len(chosen),
                "max_degree": max(deg.values()) if deg else 0,
            },
        )

    async def aclose(self) -> None:
        return None
