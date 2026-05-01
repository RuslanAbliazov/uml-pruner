"""Serialize a UML node into a compact text document for embedding.

Strategy: graph-aware. The text representation of a class consists of these
sections, in fixed order:

    1. Header — short name, FQCN, type, package.
    2. Methods — filtered (drop accessors and constructors-of-self), without
       parameters, deduped.
    3. Fields — `params` from the JSON, with cache-line padding (`pNN`) and
       synthetic ($-named) members removed.
    4. Per-relation blocks — one **separate** block for every combination of
       relation type (Inheritance / Association / Dependency / ...) and
       direction (outgoing / incoming). Section names:

           Inheritance: extends/implements        (outgoing Inheritance)
           Inheritance: extended/implemented by   (incoming Inheritance)
           Association: outgoing
           Association: incoming
           Dependency: outgoing
           Dependency: incoming
           <Other>: outgoing / incoming           (any unexpected relation)

       Each block lists short class names, ranked by edge multiplicity.

Only short class names are emitted in the relations sections (the FQCN of the
node is in the header; surrounding short names give the LLM/embedder enough
semantic context without blowing up token usage).

The `description` field of nodes is intentionally NOT included: descriptions
in the source diagrams are LLM-generated and would leak ground-truth into the
retrieval signal.

Limits
------
Every limit parameter accepts ``None`` to mean "no limit" — the section will
contain every method / field / neighbor without truncation. The CLI / YAML
config translates ``-1`` (or any negative integer) and ``null`` to ``None``;
inside this module we always use ``None``.

Edges are expected to be in the normalized form produced by
`scripts/normalize_diagrams.py`:

    {"node_id_from": ..., "node_id_to": ..., "description": ...}

where `description` is one of: "Inheritance", "Association", "Dependency".
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Optional

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _short_name(node_id: str) -> str:
    """Return the short class name (last segment of FQCN)."""
    return node_id.rsplit(".", 1)[-1] if "." in node_id else node_id


def _package_of(node_id: str) -> str:
    """Return the package portion (everything before the short name).

    Inner-class FQCNs like `pkg.Outer.Inner` collapse to `pkg.Outer` here,
    which is fine for embedding context (we just want the locality).
    """
    return node_id.rsplit(".", 1)[0] if "." in node_id else ""


_ACCESSOR_PREFIXES = ("get", "set", "is", "has")


def _is_padding_name(name: str) -> bool:
    """Heuristic: detect cache-line padding fields like `p10`, `p17`, `p126`.

    These are deliberately content-free and add noise to embeddings.
    """
    if len(name) < 2 or name[0] != "p":
        return False
    return name[1:].isdigit()


def _clean_method(method: str, owner_short: str) -> str:
    """Normalize a method signature for embedding text.

    `public foo(int a, String b)` -> `public foo`.
    Returns "" if this method is uninformative (accessor/constructor of self).
    """
    name_part = method.split("(")[0].strip()
    if not name_part:
        return ""
    # last token is the method name (possibly preceded by visibility/modifiers)
    tokens = name_part.split()
    bare = tokens[-1] if tokens else name_part

    # drop constructor-of-self
    if bare == owner_short:
        return ""
    # drop simple accessors (very common; near-zero semantic value)
    lower = bare.lower()
    for pref in _ACCESSOR_PREFIXES:
        if lower.startswith(pref) and len(bare) > len(pref):
            return ""
    return name_part


# ----------------------------------------------------------------------------
# Edge index — built once per diagram, reused for every node
# ----------------------------------------------------------------------------


# Order of relation kinds in the output: stronger semantics first. Kinds not
# listed here are emitted after these in alphabetical order (catch-all).
_RELATION_ORDER = ("Inheritance", "Association", "Dependency")


# Section header per (relation, direction) combo. Inheritance gets bespoke
# wording because "extends/implements" reads better than "outgoing inheritance".
_SECTION_HEADERS = {
    ("Inheritance", "out"): "Inheritance: Extends/implements",
    ("Inheritance", "in"): "Inheritance: Extended/implemented by",
    ("Association", "out"): "Association: To have as a part",
    ("Association", "in"): "Association: Part of",
    ("Dependency", "out"): "Dependency: Uses",
    ("Dependency", "in"): "Dependency: Used by",
}


def _section_header(rel: str, direction: str) -> str:
    """Header for a relation block; falls back to a sensible default for any
    unexpected relation kind."""
    if (rel, direction) in _SECTION_HEADERS:
        return _SECTION_HEADERS[(rel, direction)]
    return f"{rel}: {'outgoing' if direction == 'out' else 'incoming'}"


class EdgeIndex:
    """Adjacency lookup, grouped by relation kind, with edge counts.

    Built once for a whole diagram in O(|edges|), then queried per node in
    O(neighbors) time when serializing.
    """

    __slots__ = ("_outgoing", "_incoming", "_relation_kinds")

    def __init__(self) -> None:
        # Maps node_id -> {relation -> Counter(other_node_id -> #edges)}
        self._outgoing: dict[str, dict[str, Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        self._incoming: dict[str, dict[str, Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        # Set of relation kinds actually seen in the diagram (used to surface
        # any unexpected kinds in the output deterministically).
        self._relation_kinds: set[str] = set()

    @classmethod
    def from_edges(cls, edges: Iterable[dict[str, Any]]) -> "EdgeIndex":
        idx = cls()
        for e in edges:
            src = e.get("node_id_from")
            dst = e.get("node_id_to")
            rel = (e.get("description") or "").strip() or "Other"
            if not src or not dst or src == dst:
                continue
            idx._outgoing[src][rel][dst] += 1
            idx._incoming[dst][rel][src] += 1
            idx._relation_kinds.add(rel)
        return idx

    def outgoing(self, node_id: str) -> dict[str, Counter]:
        return self._outgoing.get(node_id, {})

    def incoming(self, node_id: str) -> dict[str, Counter]:
        return self._incoming.get(node_id, {})

    def relation_kinds(self) -> list[str]:
        """Return the relation kinds in canonical order:
        the well-known ones first (Inheritance, Association, Dependency),
        then any other kinds in alphabetical order (deterministic).
        """
        known = [k for k in _RELATION_ORDER if k in self._relation_kinds]
        others = sorted(self._relation_kinds - set(_RELATION_ORDER))
        return known + others


# ----------------------------------------------------------------------------
# Relation rendering
# ----------------------------------------------------------------------------


def _format_neighbor_list(
    counter: Counter,
    max_items: Optional[int],
) -> list[str]:
    """Return a list of short names for the top-N neighbors by edge count.

    `max_items=None` → no truncation (every neighbor is emitted).
    """
    if not counter:
        return []
    # Sort by (count desc, node_id asc) for determinism.
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))

    out: list[str] = []
    truncated = False
    for nid, n in ranked:
        if max_items is not None and len(out) >= max_items:
            truncated = True
            break
        short = _short_name(nid)
        if n > 1:
            out.append(f"{short} (x{n})")
        else:
            out.append(short)

    if truncated:
        remaining = len(ranked) - len(out)
        out.append(f"... +{remaining} more")
    return out


def _emit_relation_block(
    lines: list[str],
    counter: Counter,
    rel: str,
    direction: str,
    max_items: Optional[int],
) -> None:
    """Append a `<header>:\n  - X\n  - Y` block to `lines` if non-empty."""
    items = _format_neighbor_list(counter, max_items)
    if not items:
        return
    lines.append(f"{_section_header(rel, direction)}:")
    for it in items:
        lines.append(f"  - {it}")


# ----------------------------------------------------------------------------
# Main entry points
# ----------------------------------------------------------------------------


def node_to_text(
    node: dict[str, Any],
    edge_index: EdgeIndex | None = None,
    *,
    max_methods: Optional[int] = 25,
    max_fields: Optional[int] = 20,
    max_outgoing_per_relation: Optional[int] = 12,
    max_incoming_per_relation: Optional[int] = 12,
) -> str:
    """Serialize a single node to text.

    Args:
        node: Node dict with at least `node_id`. May include `type`, `methods`,
            `params`.
        edge_index: Pre-built EdgeIndex over the whole diagram. If None, the
            relations sections are omitted (backwards-compatible mode).
        max_methods: Max methods listed; ``None`` for unlimited.
        max_fields: Max fields listed; ``None`` for unlimited.
        max_outgoing_per_relation: Per-block cap for outgoing relations
            (Inheritance / Association / Dependency / others). ``None`` for
            unlimited.
        max_incoming_per_relation: Per-block cap for incoming relations.
            ``None`` for unlimited.

    Returns:
        Multi-line string. Stable across runs.
    """
    node_id = node.get("node_id", "")
    short = _short_name(node_id)
    pkg = _package_of(node_id)
    node_type = node.get("type", "class")

    lines: list[str] = [
        f"Name: {short}",
        f"FQCN: {node_id}",
        f"Type: {node_type}",
    ]
    if pkg:
        lines.append(f"Package: {pkg}")

    # ---- Methods --------------------------------------------------------
    # Dedupe by visibility+name to collapse overloads (e.g. publishEvent(...)
    # variants that differ only by parameter list).
    methods = node.get("methods") or []
    cleaned_methods: list[str] = []
    seen_methods: set[str] = set()
    for m in methods:
        c = _clean_method(m, short)
        if not c or c in seen_methods:
            continue
        seen_methods.add(c)
        cleaned_methods.append(c)
    if cleaned_methods:
        lines.append("Methods:")
        head = (
            cleaned_methods if max_methods is None else cleaned_methods[:max_methods]
        )
        for m in head:
            lines.append(f"  - {m}")
        if max_methods is not None and len(cleaned_methods) > max_methods:
            lines.append(f"  - ... +{len(cleaned_methods) - max_methods} more")

    # ---- Fields ---------------------------------------------------------
    # Filter likely-noise: padding fields (`pNN`) common in lock-free code,
    # and synthetic `$`-named members.
    fields = node.get("params") or []
    cleaned_fields: list[str] = []
    for f in fields:
        cleaned = (f or "").strip()
        if not cleaned:
            continue
        # the field name is the last whitespace-separated token (after type)
        last = cleaned.rsplit(" ", 1)[-1]
        if "$" in last:
            continue
        if _is_padding_name(last):
            continue
        cleaned_fields.append(cleaned)
    if cleaned_fields:
        lines.append("Fields:")
        head = cleaned_fields if max_fields is None else cleaned_fields[:max_fields]
        for f in head:
            lines.append(f"  - {f}")
        if max_fields is not None and len(cleaned_fields) > max_fields:
            lines.append(f"  - ... +{len(cleaned_fields) - max_fields} more")

    # ---- Relations ------------------------------------------------------
    # One block per (relation_kind, direction), in canonical order.
    if edge_index is not None:
        out_rels = edge_index.outgoing(node_id)
        in_rels = edge_index.incoming(node_id)
        # Union of relation kinds present on this node, ordered canonically.
        kinds_here = list(_RELATION_ORDER)
        for k in sorted(set(out_rels) | set(in_rels)):
            if k not in kinds_here:
                kinds_here.append(k)

        for rel in kinds_here:
            _emit_relation_block(
                lines,
                out_rels.get(rel, Counter()),
                rel,
                "out",
                max_outgoing_per_relation,
            )
            _emit_relation_block(
                lines,
                in_rels.get(rel, Counter()),
                rel,
                "in",
                max_incoming_per_relation,
            )

    return "\n".join(lines)


def nodes_to_texts(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]] | None = None,
    *,
    max_methods: Optional[int] = None,
    max_fields: Optional[int] = None,
    max_outgoing_per_relation: Optional[int] = None,
    max_incoming_per_relation: Optional[int] = None,
) -> list[str]:
    """Serialize many nodes. Order is preserved.

    `edges` should be the normalized edge list (the output of
    `scripts/normalize_diagrams.py`). If omitted, the function falls back to
    the old SELF-CONTAINED format.

    Any limit may be ``None`` to disable truncation.
    """
    edge_index = EdgeIndex.from_edges(edges) if edges else None
    return [
        node_to_text(
            n,
            edge_index=edge_index,
            max_methods=max_methods,
            max_fields=max_fields,
            max_outgoing_per_relation=max_outgoing_per_relation,
            max_incoming_per_relation=max_incoming_per_relation,
        )
        for n in nodes
    ]
