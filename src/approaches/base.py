"""Common interface every UML-pruning approach must satisfy.

Goal: let the benchmark and any CLI consume different approaches uniformly,
without knowing how they work internally. Each approach takes the same input
shape (a query + a normalized diagram) and emits the same output shape (a
pruned subgraph + standard metadata).

Conventions
-----------

* All approaches are async (the LLM-backed ones do parallel calls).
* Approaches do NOT load the dataset or write files themselves; they receive
  inputs in memory and return a result. Persistence is handled by callers
  (e.g. ``scripts/benchmark.py``).
* The diagram passed in is the *normalized* one produced by
  ``scripts/normalize_diagrams.py``: edges only carry
  ``(node_id_from, node_id_to, description)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ApproachInputs:
    """Inputs every approach receives for a single sample.

    DESIGN INVARIANT — *no ground-truth leakage*:

    In production, an approach only ever knows the user's free-form query
    and the full normalized diagram. It must NOT see the focus / golden
    class. To make that impossible by construction, this dataclass
    deliberately omits ``central_node`` and the per-sample annotations.

    ``sample_id`` and ``repo`` are kept ONLY for two operational reasons:
        * ``sample_id`` — output filenames + correlating logs with errors.
        * ``repo``      — locating the matching pre-built embedding index
                          on disk (``data/embeddings/<repo_basename>``).
    Approaches must not derive any decision from them.
    """

    query: str
    diagram: dict[str, Any]
    sample_id: str = ""
    repo: str = ""

    @property
    def nodes(self) -> list[dict[str, Any]]:
        return self.diagram.get("nodes", [])

    @property
    def edges(self) -> list[dict[str, Any]]:
        return self.diagram.get("edges", [])


@dataclass
class ApproachResult:
    """Standard result every approach returns.

    Fields:
        nodes / edges     : the pruned subgraph in the same {nodes, edges}
                            shape as the input diagram.
        required_node_ids : node_ids the approach considers strictly required.
        useful_node_ids   : node_ids the approach considers supporting/useful.
        metadata          : free-form diagnostic info: timings, batch sizes,
                            LLM token usage, intermediate stage counts, etc.
                            Always serializable to JSON.
        approach          : name of the approach that produced this result.
    """

    approach: str
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    required_node_ids: list[str] = field(default_factory=list)
    useful_node_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_diagram(self) -> dict[str, Any]:
        """Return a JSON-serializable dict in the project's standard
        ``{nodes, edges, metadata}`` shape used by evaluator and pipeline."""
        meta = dict(self.metadata)
        meta.setdefault("approach", self.approach)
        meta.setdefault("required_node_ids", sorted(self.required_node_ids))
        meta.setdefault("useful_node_ids", sorted(self.useful_node_ids))
        meta.setdefault("filtered_node_count", len(self.nodes))
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "metadata": meta,
        }


class ApproachRunner(Protocol):
    """Every approach implements this."""

    name: str

    async def run(self, inputs: ApproachInputs) -> ApproachResult:  # pragma: no cover
        """Produce a pruned subgraph + classifications for the input sample."""
        ...

    async def aclose(self) -> None:  # pragma: no cover
        """Release any resources (LLM clients, encoder handles, etc).

        The benchmark calls this once when it's done iterating samples.
        """
        ...
