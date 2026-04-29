"""Serialize a UML node into a compact text document for embedding.

Policy: SELF-CONTAINED — we include only information intrinsic to the node
itself (node_id, type, description, methods). Connections / edges are NOT
included. This keeps the representation deterministic and independent of the
graph, and avoids contamination from neighbor descriptions.

If at some point we want richer context (neighbors), add a separate
function rather than mixing concerns here.
"""

from __future__ import annotations

from typing import Any


def _short_name(node_id: str) -> str:
    """Return the short class name (last segment of FQCN)."""
    return node_id.rsplit(".", 1)[-1] if "." in node_id else node_id


def _clean_method(method: str) -> str:
    """Drop parameter types: `public foo(int a, String b)` -> `public foo`.

    Keeps visibility + method name. Parameter names/types rarely help embedding
    semantics and add a lot of noise.
    """
    name_part = method.split("(")[0].strip()
    return name_part


def node_to_text(
    node: dict[str, Any],
    max_methods: int = 30,
    max_description_chars: int = 4000,
    max_fields: int = 30,
) -> str:
    """Serialize a single node into a text document suitable for embedding.

    Args:
        node: Node dict with at least 'node_id'. May also have 'type',
              'description', 'methods', 'params'.
        max_methods: Truncate method list to this many entries.
        max_description_chars: Truncate description to this many chars.
        max_fields: Truncate field list to this many entries.

    Returns:
        A multi-line string representation. Stable across runs.

    Example output:
        Name: ExternalLocation
        FQCN: ghidra.program.model.symbol.ExternalLocation
        Type: interface
        Description: Represents a location in an external program (library).
        Methods:
          - getExternalProgramName
          - getLabel
          - getAddress
        Fields:
          - private static final Logger log
          - private SentenceGenerator sentenceGenerator
    """
    node_id = node.get("node_id", "")
    short = _short_name(node_id)
    node_type = node.get("type", "class")

    lines: list[str] = [
        f"Name: {short}",
        f"FQCN: {node_id}",
        f"Type: {node_type}",
    ]

    # it is unfair to put description to RAG
    # description = (node.get("description") or "").strip()
    # if description:
    #     if len(description) > max_description_chars:
    #         description = description[: max_description_chars - 3] + "..."
    #     # collapse internal newlines to keep the doc compact
    #     description = " ".join(description.split())
    #     lines.append(f"Description: {description}")

    methods = node.get("methods") or []
    i = 0
    if methods:
        lines.append("Methods:")
        for m in methods:
            cleaned = _clean_method(m)
            if "get" in cleaned.lower() or "set" in cleaned.lower():
                continue

            if cleaned == short:
                continue

            if cleaned:
                lines.append(f"  - {cleaned}")
                i += 1

            if i > max_methods:
                lines.append(f"  - ...  more)")

    fields = node.get("params") or []
    if fields:
        lines.append("Fields:")
        for f in fields[:max_fields]:
            cleaned = f.strip()
            if cleaned:
                lines.append(f"  - {cleaned}")
        if len(fields) > max_fields:
            lines.append(f"  - ... more)")

    return "\n".join(lines)


def nodes_to_texts(
    nodes: list[dict[str, Any]],
    max_methods: int = 30,
    max_description_chars: int = 4000,
    max_fields: int = 30,
) -> list[str]:
    """Serialize many nodes. Order is preserved."""
    return [
        node_to_text(
            n, max_methods=max_methods, max_description_chars=max_description_chars, max_fields=max_fields
        )
        for n in nodes
    ]
