"""Oracle baselines — predictions that USE ground truth.

These are NOT approaches: they intentionally violate the
"runner sees only (query, diagram)" invariant. Their job is to give us
upper bounds and sanity checks against which any real approach can be
compared.

Architecturally they live under ``src.eval`` (which is already
ground-truth-aware) rather than under ``src.approaches`` (which is not).

Each function takes ``(sample, diagram)`` and returns the set of predicted
node_ids. The caller (``scripts/run_oracle_baselines.py``) builds the
standard ``{nodes, edges, metadata}`` JSON and writes it next to the real
approaches' results, so the same evaluator works on both.

Available oracles:

    * :func:`predict_central_plus_neighbors` — central_node + 1-hop
    * :func:`predict_gold_only`              — exactly required ∪ useful
"""

from __future__ import annotations

from typing import Any, Iterable

from src.eval.annotations import AnnotationSample


def _direct_neighbours(node_id: str, edges: Iterable[dict[str, Any]]) -> set[str]:
    """All node_ids connected to ``node_id`` by at least one edge.

    Self-loops contribute nothing. Direction and relation kind are ignored.
    """
    out: set[str] = set()
    for e in edges:
        a = e.get("node_id_from")
        b = e.get("node_id_to")
        if not a or not b or a == b:
            continue
        if a == node_id and b != node_id:
            out.add(b)
        elif b == node_id and a != node_id:
            out.add(a)
    return out


def predict_central_plus_neighbors(
    sample: AnnotationSample,
    diagram: dict[str, Any],
) -> set[str]:
    """Oracle: central_node ∪ {its direct neighbours in the diagram}.

    Use this to estimate the recall ceiling of any "anchor + 1-hop"
    architecture. If even this oracle stops well below F1 = 1.0, no
    approach that follows the same shape can reach higher.

    If ``central_node`` isn't actually a node in the diagram (rare but
    possible after normalization mismatches), returns just ``{central_node}``
    so the predicted set is never empty for a present sample.
    """
    central = sample.central_node
    if not central:
        return set()

    valid_ids: set[str] = {
        n["node_id"] for n in diagram.get("nodes", []) if n.get("node_id")
    }
    if central not in valid_ids:
        return {central}

    neighbours = _direct_neighbours(central, diagram.get("edges", []))
    # Constrain to ids that actually exist in the diagram.
    neighbours &= valid_ids
    return {central} | neighbours


def predict_gold_only(
    sample: AnnotationSample,
    diagram: dict[str, Any],  # noqa: ARG001 — kept for uniform signature
) -> set[str]:
    """Oracle: predict exactly required ∪ useful from the gold annotations.

    F1 should always be 1.0 against the same dataset; if it isn't, the
    evaluator is misreading something. Cheap sanity check.
    """
    return {nid for nid, lbl in sample.annotations.items() if lbl in ("required", "useful")}


# Registry mapping CLI name -> prediction function. Adding a new oracle is
# just a new entry here.
ORACLES = {
    "central_plus_neighbors": predict_central_plus_neighbors,
    "gold_only": predict_gold_only,
}
