"""Unit tests for oracle baselines."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.annotations import AnnotationSample  # noqa: E402
from src.eval.oracle_baselines import (  # noqa: E402
    ORACLES,
    predict_central_plus_neighbors,
    predict_gold_only,
)


def _diagram() -> dict:
    return {
        "nodes": [
            {"node_id": "p.A"},
            {"node_id": "p.B"},
            {"node_id": "p.C"},
            {"node_id": "p.D"},
            {"node_id": "p.E"},
        ],
        "edges": [
            {"node_id_from": "p.A", "node_id_to": "p.B", "description": "Association"},
            {"node_id_from": "p.A", "node_id_to": "p.C", "description": "Dependency"},
            {"node_id_from": "p.B", "node_id_to": "p.D", "description": "Association"},
            {"node_id_from": "p.D", "node_id_to": "p.E", "description": "Inheritance"},
        ],
    }


def _sample(central: str, annotations: dict[str, str]) -> AnnotationSample:
    return AnnotationSample(
        _id="x",
        sample_id="s",
        task_id="t",
        central_node=central,
        repo="test/tiny",
        query="q",
        annotations=annotations,
    )


def test_central_plus_neighbors_picks_one_hop():
    """A's direct neighbours are B and C (out-edges); D and E are 2 hops away."""
    s = _sample("p.A", {"p.A": "required", "p.B": "useful"})
    pred = predict_central_plus_neighbors(s, _diagram())
    assert pred == {"p.A", "p.B", "p.C"}


def test_central_plus_neighbors_includes_incoming_edges():
    """B receives an edge from A — even though B is the 'central' here, A must
    appear because we pick neighbours regardless of edge direction."""
    s = _sample("p.B", {"p.B": "required"})
    pred = predict_central_plus_neighbors(s, _diagram())
    # B's neighbours: A (incoming Association) and D (outgoing Association)
    assert pred == {"p.B", "p.A", "p.D"}


def test_central_plus_neighbors_handles_unknown_central():
    """If central_node isn't in the diagram, we still return at least itself —
    so downstream code doesn't crash on empty predictions."""
    s = _sample("p.NOT_HERE", {})
    pred = predict_central_plus_neighbors(s, _diagram())
    assert pred == {"p.NOT_HERE"}


def test_central_plus_neighbors_skips_self_loops():
    diag = _diagram()
    diag["edges"].append(
        {"node_id_from": "p.A", "node_id_to": "p.A", "description": "Dependency"}
    )
    s = _sample("p.A", {})
    pred = predict_central_plus_neighbors(s, diag)
    # Self-loop must NOT add p.A again as a "neighbour" — predicted set is
    # central_node + true neighbours.
    assert pred == {"p.A", "p.B", "p.C"}


def test_central_plus_neighbors_empty_central_returns_empty():
    s = _sample("", {})
    assert predict_central_plus_neighbors(s, _diagram()) == set()


def test_gold_only_recovers_required_and_useful():
    s = _sample("p.A", {"p.A": "required", "p.B": "useful", "p.X": "irrelevant"})
    assert predict_gold_only(s, _diagram()) == {"p.A", "p.B"}


def test_gold_only_with_no_annotations():
    s = _sample("p.A", {})
    assert predict_gold_only(s, _diagram()) == set()


def test_oracles_registry_is_complete():
    assert set(ORACLES) == {"central_plus_neighbors", "gold_only"}
