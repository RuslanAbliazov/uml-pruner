"""Build compact per-class representations for Stage 2."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


CompactLevel = str  # "full" | "compact" | "ultra"

_COMPACT_PRESETS: dict[str, dict[str, Any]] = {
    # Full: used by Stage 3 (but note Stage 3 also often uses original nodes
    # directly, not this function).
    "full": {
        "max_methods": 10,
        "max_connections": 12,
        "description_max_chars": 300,
        "keep_description": True,
    },
    # Compact: used by Stage 2 and by Stage 3 under pressure.
    "compact": {
        "max_methods": 5,
        "max_connections": 8,
        "description_max_chars": 150,
        "keep_description": True,
    },
    # Ultra: used when a single item is still too big for budget. Drops most
    # optional information.
    "ultra": {
        "max_methods": 2,
        "max_connections": 3,
        "description_max_chars": 0,
        "keep_description": False,
    },
}


def build_class_representation(
    node: dict[str, Any],
    outgoing_edges: list[dict[str, Any]],
    incoming_edges: list[dict[str, Any]],
    compact_level: CompactLevel = "compact",
    max_methods: int | None = None,
    max_connections: int | None = None,
) -> dict[str, Any]:
    """Create a compact dict representation of a class.

    Args:
        node: Original node dict.
        outgoing_edges: Edges where node_id == this node's id.
        incoming_edges: Edges where node_id_to == this node's id.
        compact_level: Preset controlling detail level:
            - "full":   max detail (used for Stage 3 candidates).
            - "compact": balanced (Stage 2 default).
            - "ultra":  minimal (oversized-element fallback).
        max_methods / max_connections: Optional overrides of the preset.
    """
    preset = _COMPACT_PRESETS.get(compact_level, _COMPACT_PRESETS["compact"])
    mm = max_methods if max_methods is not None else preset["max_methods"]
    mc = max_connections if max_connections is not None else preset["max_connections"]
    desc_max = preset["description_max_chars"]
    keep_desc = preset["keep_description"]

    # Methods: keep "visibility methodName" without param types.
    methods = node.get("methods", [])
    shortened: list[str] = []
    for m in methods[:mm]:
        name_part = m.split("(")[0].strip()
        shortened.append(name_part)

    extends: list[str] = []
    implements: list[str] = []
    uses: list[str] = []
    for e in outgoing_edges[: mc * 3]:
        target = e.get("node_id_to", "")
        desc = e.get("description", "")
        sub = e.get("subdescription", "")
        if desc == "Inheritance":
            if sub == "Extends":
                extends.append(target)
            elif sub == "Implements":
                implements.append(target)
        elif desc in ("Association", "Dependency"):
            uses.append(target)

    used_by: list[str] = []
    for e in incoming_edges[:mc]:
        source = e.get("node_id_from", "")
        used_by.append(source)

    extends = extends[:mc]
    implements = implements[:mc]
    uses = list(dict.fromkeys(uses))[:mc]
    used_by = list(dict.fromkeys(used_by))[:mc]

    representation: dict[str, Any] = {
        "node_id": node["node_id"],
        "type": node.get("type", "class"),
    }
    if shortened:
        representation["methods"] = shortened

    connections: dict[str, list[str]] = {}
    if extends:
        connections["extends"] = extends
    if implements:
        connections["implements"] = implements
    if uses:
        connections["uses"] = uses
    if used_by:
        connections["used_by"] = used_by
    if connections:
        representation["connections"] = connections

    if keep_desc and desc_max > 0:
        desc = node.get("description", "") or ""
        if desc:
            if len(desc) <= desc_max:
                representation["description"] = desc
            else:
                representation["description"] = desc[: desc_max - 3] + "..."

    return representation


def build_edge_index(
    edges: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Build outgoing / incoming edge indices keyed by node_id."""
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in edges:
        outgoing[e["node_id_from"]].append(e)
        incoming[e["node_id_to"]].append(e)
    return dict(outgoing), dict(incoming)


def filter_subgraph(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    node_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return nodes with id in node_ids and edges whose both endpoints are kept."""
    filtered_nodes = [n for n in nodes if n["node_id"] in node_ids]
    filtered_edges = [
        e
        for e in edges
        if e.get("node_id_from") in node_ids and e.get("node_id_to") in node_ids
    ]
    return filtered_nodes, filtered_edges
