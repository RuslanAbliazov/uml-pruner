"""Этап 3 — собрать 1-hop окрестность вокруг anchor.

Алгоритм:

* Сосед — это любой узел, связанный с anchor хотя бы одним ребром.
  Направление ребра (`from`/`to`) и тип связи (Inheritance / Association /
  Dependency / ...) НЕ важны: даже зависимости `used_by` тянем.
* Self-loops отбрасываем — они не дают новой информации.
* Если соседей много (например, anchor — это `StringUtils` с 500 ссылок),
  применяем потолок `cap`: оставляем anchor + top-(cap-1) соседей с
  максимальным числом рёбер к anchor (та же связь по нескольким причинам
  → выше ранг). Тай-брейк по node_id для детерминизма.

Метрики потом считаются по объединению `{anchor} ∪ оставленные соседи`.
"""

from __future__ import annotations

from typing import Any

from src.approaches.anchor_neighbors.stage_outputs import StageName, StageOutcome


def _direct_neighbors(anchor: str, edges: list[dict[str, Any]]) -> set[str]:
    """Все узлы, связанные с anchor хотя бы одним ребром (без self-loops)."""
    out: set[str] = set()
    for e in edges:
        a = e.get("node_id_from")
        b = e.get("node_id_to")
        if a == anchor and b and b != anchor:
            out.add(b)
        elif b == anchor and a and a != anchor:
            out.add(a)
    return out


def _trim_by_edge_multiplicity(
    *,
    anchor: str,
    neighbors: set[str],
    edges: list[dict[str, Any]],
    cap: int,
) -> set[str]:
    """Если соседей больше потолка — оставляем cap-1 самых «крепко связанных».

    Считаем кол-во рёбер к anchor у каждого соседа; чем больше — тем
    важнее структурно. Сам anchor всегда сохраняется."""
    multiplicity: dict[str, int] = {}
    for e in edges:
        a = e.get("node_id_from")
        b = e.get("node_id_to")
        if a == anchor and b in neighbors:
            multiplicity[b] = multiplicity.get(b, 0) + 1
        elif b == anchor and a in neighbors:
            multiplicity[a] = multiplicity.get(a, 0) + 1
    ranked = sorted(neighbors, key=lambda n: (-multiplicity.get(n, 0), n))
    return {anchor, *ranked[: max(0, cap - 1)]}


def expand_neighbors(
    *,
    anchor: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    cap: int,
) -> StageOutcome:
    """Собрать подграф `anchor + 1-hop соседи` (с опциональным потолком).

    Возвращаем `StageOutcome`, в payload которого лежат материалы для
    этапа 4: `sub_nodes`, `sub_edges`. `node_ids` — это полное множество
    узлов подграфа (используется для оценки coverage этапа).
    """
    raw_neighbors = _direct_neighbors(anchor, edges)
    subgraph_ids: set[str] = {anchor} | raw_neighbors

    truncated = False
    if cap and len(subgraph_ids) > cap:
        truncated = True
        subgraph_ids = _trim_by_edge_multiplicity(
            anchor=anchor, neighbors=raw_neighbors, edges=edges, cap=cap,
        )

    # Узлы и рёбра подграфа: рёбра остаются только если оба конца в подграфе.
    sub_nodes = [node_by_id[nid] for nid in subgraph_ids if nid in node_by_id]
    sub_edges = [
        e for e in edges
        if e.get("node_id_from") in subgraph_ids
        and e.get("node_id_to") in subgraph_ids
    ]

    return StageOutcome(
        stage=StageName.NEIGHBORS,
        node_ids=sorted(subgraph_ids),
        payload={
            "anchor": anchor,
            "sub_nodes": sub_nodes,
            "sub_edges": sub_edges,
        },
        info={
            "neighbors_total": len(raw_neighbors),
            "subgraph_size": len(subgraph_ids),
            "truncated": truncated,
            "cap": cap,
        },
    )
