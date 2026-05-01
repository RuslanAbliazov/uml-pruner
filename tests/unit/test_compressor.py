"""Unit tests for build_class_representation and compact_level presets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.approaches._common.compressor import build_class_representation


def _node_with_many_methods(n: int) -> dict:
    return {
        "node_id": "com.example.Foo",
        "type": "class",
        "name": "com.example.Foo",
        "methods": [f"public method{i}(int a, String b)" for i in range(n)],
        "description": "A very long description " * 50,
    }


def _edges(count: int, kind: str = "uses"):
    if kind == "uses":
        return [
            {
                "node_id_from": "com.example.Foo",
                "node_id_to": f"com.example.Other{i}",
                "description": "Association",
                "subdescription": "HAS",
            }
            for i in range(count)
        ]
    if kind == "incoming":
        return [
            {
                "node_id_from": f"com.example.Caller{i}",
                "node_id_to": "com.example.Foo",
                "description": "Dependency",
                "subdescription": "USES",
            }
            for i in range(count)
        ]
    raise ValueError(kind)


def test_compact_has_methods_and_description():
    node = _node_with_many_methods(10)
    rep = build_class_representation(node, [], [], compact_level="compact")
    assert rep["node_id"] == "com.example.Foo"
    assert len(rep["methods"]) == 5  # compact max_methods
    assert "description" in rep  # kept
    assert len(rep["description"]) <= 150


def test_full_allows_more_detail():
    node = _node_with_many_methods(20)
    rep = build_class_representation(node, [], [], compact_level="full")
    assert len(rep["methods"]) == 10
    assert len(rep.get("description", "")) <= 300


def test_ultra_drops_description():
    node = _node_with_many_methods(10)
    rep = build_class_representation(node, [], [], compact_level="ultra")
    assert len(rep["methods"]) == 2
    assert "description" not in rep


def test_ultra_is_smaller_than_compact():
    node = _node_with_many_methods(10)
    outgoing = _edges(15)
    incoming = _edges(15, "incoming")

    compact = build_class_representation(
        node, outgoing, incoming, compact_level="compact"
    )
    ultra = build_class_representation(node, outgoing, incoming, compact_level="ultra")

    size_compact = len(json.dumps(compact))
    size_ultra = len(json.dumps(ultra))
    assert size_ultra < size_compact, (
        f"ultra should be smaller: ultra={size_ultra} compact={size_compact}"
    )


def test_connections_trimmed_to_max():
    node = {"node_id": "A", "type": "class", "methods": [], "description": ""}
    outgoing = _edges(20)
    rep = build_class_representation(node, outgoing, [], compact_level="compact")
    assert len(rep["connections"]["uses"]) == 8  # compact max_connections


def test_explicit_overrides_preset():
    node = _node_with_many_methods(10)
    rep = build_class_representation(
        node, [], [], compact_level="compact", max_methods=3, max_connections=2
    )
    assert len(rep["methods"]) == 3


def run_all():
    tests = [
        test_compact_has_methods_and_description,
        test_full_allows_more_detail,
        test_ultra_drops_description,
        test_ultra_is_smaller_than_compact,
        test_connections_trimmed_to_max,
        test_explicit_overrides_preset,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")


if __name__ == "__main__":
    run_all()
