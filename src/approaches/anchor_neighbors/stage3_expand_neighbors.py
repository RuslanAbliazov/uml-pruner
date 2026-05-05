"""Этап 3 — собрать 1-hop окрестность вокруг каждого якоря.

Алгоритм:

* Сосед — это любой узел, связанный с anchor хотя бы одним ребром.
  Направление ребра (`from`/`to`) и тип связи (Inheritance / Association /
  Dependency / ...) НЕ важны: даже зависимости `used_by` тянем.
* Self-loops отбрасываем — они не дают новой информации.
* Multi-anchor режим: соседи считаются для КАЖДОГО якоря, результаты
  объединяются. Все якоря сами по себе всегда остаются в подграфе.
* Если суммарный подграф больше потолка `cap`, обрезаем по «структурной
  важности» — каждому соседу присваиваем кол-во рёбер к ЛЮБОМУ из якорей,
  оставляем top-(cap - len(anchors)) самых связных. Тай-брейк по node_id
  для детерминизма.

Метрики потом считаются по объединению `anchors ∪ оставленные соседи`.

Бэк-совместимость: при ``len(anchors) == 1`` поведение совпадает с
прежним подходом «один anchor».
"""

from __future__ import annotations

from typing import Any, Iterable

from src.approaches.anchor_neighbors.stage_outputs import StageName, StageOutcome


def _direct_neighbors(
    anchors: set[str], edges: list[dict[str, Any]]
) -> set[str]:
    """Все узлы, связанные хотя бы одним ребром с любым из anchors.

    Сами якоря из множества соседей исключаем — они и так в подграфе.
    Self-loops тоже отбрасываются.
    """
    out: set[str] = set()
    for e in edges:
        a = e.get("node_id_from")
        b = e.get("node_id_to")
        if not a or not b or a == b:
            continue
        if a in anchors and b not in anchors:
            out.add(b)
        elif b in anchors and a not in anchors:
            out.add(a)
    return out


def _trim_by_edge_multiplicity(
    *,
    anchors: set[str],
    neighbors: set[str],
    edges: list[dict[str, Any]],
    budget: int,
) -> set[str]:
    """Из ``neighbors`` оставить top-``budget`` самых связных с якорями.

    Связность = сколько рёбер у соседа к ЛЮБОМУ якорю. Тай-брейк по
    node_id для детерминизма. Сами якоря наверх не возвращаются — этим
    занимается вызывающий.
    """
    if budget <= 0:
        return set()
    multiplicity: dict[str, int] = {}
    for e in edges:
        a = e.get("node_id_from")
        b = e.get("node_id_to")
        if not a or not b or a == b:
            continue
        if a in anchors and b in neighbors:
            multiplicity[b] = multiplicity.get(b, 0) + 1
        elif b in anchors and a in neighbors:
            multiplicity[a] = multiplicity.get(a, 0) + 1
    ranked = sorted(neighbors, key=lambda n: (-multiplicity.get(n, 0), n))
    return set(ranked[:budget])


def expand_neighbors(
    *,
    anchors: Iterable[str],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    cap: int,
) -> StageOutcome:
    """Собрать подграф `anchors ∪ их 1-hop соседи` (с опциональным потолком).

    Возвращаем `StageOutcome`, в payload которого лежат материалы для
    этапа 4: `sub_nodes`, `sub_edges`. `node_ids` — это полное множество
    узлов подграфа (используется для оценки coverage этапа).
    """
    # Нормализуем вход: убираем пустые/дубликаты, сохраняем порядок (это
    # пригодится для top-1-первого в payload['anchor']).
    anchor_list: list[str] = []
    seen: set[str] = set()
    for a in anchors:
        if a and a not in seen:
            anchor_list.append(a)
            seen.add(a)
    anchor_set: set[str] = set(anchor_list)

    raw_neighbors = _direct_neighbors(anchor_set, edges)
    subgraph_ids: set[str] = anchor_set | raw_neighbors

    truncated = False
    # Если задан потолок и подграф его превышает — режем НЕЯКОРНУЮ часть.
    # Якоря всегда сохраняются: они «по построению» релевантны.
    if cap and len(subgraph_ids) > cap:
        truncated = True
        budget = max(0, cap - len(anchor_set))
        kept_neighbors = _trim_by_edge_multiplicity(
            anchors=anchor_set,
            neighbors=raw_neighbors,
            edges=edges,
            budget=budget,
        )
        subgraph_ids = anchor_set | kept_neighbors

    # Узлы и рёбра подграфа: рёбра остаются только если оба конца в подграфе.
    sub_nodes = [node_by_id[nid] for nid in subgraph_ids if nid in node_by_id]
    sub_edges = [
        e for e in edges
        if e.get("node_id_from") in subgraph_ids
        and e.get("node_id_to") in subgraph_ids
    ]

    # `anchor` (top-1) держим в payload для обратной совместимости —
    # старые потребители (метрики, debug-отчёт) могут читать одно поле,
    # а multi-anchor код — поле `anchors`.
    primary = anchor_list[0] if anchor_list else ""

    return StageOutcome(
        stage=StageName.NEIGHBORS,
        node_ids=sorted(subgraph_ids),
        payload={
            "anchor": primary,
            "anchors": list(anchor_list),
            "sub_nodes": sub_nodes,
            "sub_edges": sub_edges,
        },
        info={
            "n_anchors": len(anchor_set),
            "neighbors_total": len(raw_neighbors),
            "subgraph_size": len(subgraph_ids),
            "truncated": truncated,
            "cap": cap,
        },
    )
