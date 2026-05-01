"""Stage 3 — collect the anchor's 1-hop neighborhood.

Every direct neighbor counts, regardless of edge direction
(``out``/``in``) or relation kind (Inheritance / Association / Dependency
/ ...). Self-loops are dropped.

If the anchor's degree explodes (think: ``utils.StringUtils`` with 500
references), we trim the neighborhood to ``cap`` nodes total, keeping
the ones with the highest *edge multiplicity* to the anchor — that
typically picks the most structurally meaningful neighbors.
"""

from __future__ import annotations

from typing import Any


def neighbors_of(anchor: str, edges: list[dict[str, Any]]) -> set[str]:
    """All nodes connected to ``anchor`` by at least one edge."""
    out: set[str] = set()
    for e in edges:
        src = e.get("node_id_from")
        dst = e.get("node_id_to")
        if src == anchor and dst and dst != anchor:
            out.add(dst)
        elif dst == anchor and src and src != anchor:
            out.add(src)
    return out


def trim_subgraph(
    *,
    anchor: str,
    neighbors: set[str],
    edges: list[dict[str, Any]],
    cap: int,
) -> set[str]:
    """Keep ``cap-1`` neighbors with the most edges to the anchor.

    The anchor itself is always retained. Ties broken by node_id for
    determinism.
    """
    scores: dict[str, int] = {}
    for e in edges:
        src = e.get("node_id_from")
        dst = e.get("node_id_to")
        if src == anchor and dst in neighbors:
            scores[dst] = scores.get(dst, 0) + 1
        elif dst == anchor and src in neighbors:
            scores[src] = scores.get(src, 0) + 1
    ranked = sorted(neighbors, key=lambda n: (-scores.get(n, 0), n))
    return {anchor, *ranked[: max(0, cap - 1)]}


def build_subgraph(
    *,
    anchor: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    cap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return ``(sub_nodes, sub_edges, info)``.

    ``info`` carries diagnostic counts the runner surfaces in metadata:
    ``neighbors_count``, ``subgraph_truncated``.
    """
    neighbor_ids = neighbors_of(anchor, edges)
    subgraph_ids: set[str] = {anchor} | neighbor_ids

    truncated = False
    if cap and len(subgraph_ids) > cap:
        truncated = True
        subgraph_ids = trim_subgraph(
            anchor=anchor, neighbors=neighbor_ids, edges=edges, cap=cap,
        )

    sub_nodes = [node_by_id[nid] for nid in subgraph_ids if nid in node_by_id]
    sub_edges = [
        e for e in edges
        if e.get("node_id_from") in subgraph_ids
        and e.get("node_id_to") in subgraph_ids
    ]
    info = {
        "neighbors_count": len(neighbor_ids),
        "subgraph_truncated": truncated,
    }
    return sub_nodes, sub_edges, info
