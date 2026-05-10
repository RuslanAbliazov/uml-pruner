"""Unit tests for the query-agnostic baselines.

These tests don't touch any external service or model, so they run in
plain CI in milliseconds. They verify the contract of each baseline:

* Output shape matches ``ApproachResult``.
* Determinism: same input → same output across runs (where applicable).
* Edge cases: empty diagram, k > |nodes|, etc.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.approaches.baselines.runner import (  # noqa: E402
    EmptyBaseline,
    FullDiagramBaseline,
    RandomSubsetBaseline,
    TopDegreeBaseline,
    _stable_seed,
)
from src.core.types import ApproachInputs  # noqa: E402


def _diagram() -> dict:
    return {
        "nodes": [
            {"node_id": "p.A", "type": "class", "methods": [], "params": []},
            {"node_id": "p.B", "type": "class", "methods": [], "params": []},
            {"node_id": "p.C", "type": "class", "methods": [], "params": []},
            {"node_id": "p.D", "type": "class", "methods": [], "params": []},
            {"node_id": "p.E", "type": "class", "methods": [], "params": []},
        ],
        "edges": [
            {"node_id_from": "p.A", "node_id_to": "p.B", "description": "Association"},
            {"node_id_from": "p.A", "node_id_to": "p.C", "description": "Dependency"},
            {"node_id_from": "p.B", "node_id_to": "p.C", "description": "Inheritance"},
            {"node_id_from": "p.B", "node_id_to": "p.D", "description": "Association"},
            {"node_id_from": "p.D", "node_id_to": "p.E", "description": "Inheritance"},
        ],
    }


def _inputs(sample_id: str = "sample_one") -> ApproachInputs:
    return ApproachInputs(
        query="test query",
        diagram=_diagram(),
        sample_id=sample_id,
        repo="test/tiny",
    )


# ---- empty -----------------------------------------------------------------


def test_empty_baseline_predicts_nothing():
    res = asyncio.run(EmptyBaseline().run(_inputs()))
    assert res.approach == "empty"
    assert res.nodes == []
    assert res.edges == []
    assert res.required_node_ids == []
    assert res.useful_node_ids == []


# ---- full_diagram ----------------------------------------------------------


def test_full_diagram_returns_everything():
    diagram = _diagram()
    res = asyncio.run(FullDiagramBaseline().run(_inputs()))
    assert {n["node_id"] for n in res.nodes} == {n["node_id"] for n in diagram["nodes"]}
    assert len(res.edges) == len(diagram["edges"])
    # required ∪ useful covers the whole diagram
    assert set(res.required_node_ids) == {n["node_id"] for n in diagram["nodes"]}
    assert res.useful_node_ids == []
    assert res.metadata["n_nodes"] == 5
    assert res.metadata["n_edges"] == 5


# ---- random_subset --------------------------------------------------------


def test_random_subset_size_respected():
    res = asyncio.run(RandomSubsetBaseline(size=3, seed=42).run(_inputs()))
    assert len(res.nodes) == 3
    assert len(res.required_node_ids) == 3
    # Edges retained only between picked nodes
    picked = set(res.required_node_ids)
    for e in res.edges:
        assert e["node_id_from"] in picked
        assert e["node_id_to"] in picked


def test_random_subset_deterministic_per_sample_id():
    a = asyncio.run(RandomSubsetBaseline(size=3, seed=42).run(_inputs("foo")))
    b = asyncio.run(RandomSubsetBaseline(size=3, seed=42).run(_inputs("foo")))
    assert a.required_node_ids == b.required_node_ids


def test_random_subset_changes_with_sample_id():
    """Different sample_ids must (almost certainly) yield different picks."""
    a = asyncio.run(RandomSubsetBaseline(size=3, seed=42).run(_inputs("foo")))
    b = asyncio.run(RandomSubsetBaseline(size=3, seed=42).run(_inputs("bar")))
    # Astronomically unlikely to be equal on a 5-node universe with size=3 (10
    # combinations), but we assert "either differ OR equal but for the right
    # structural reason" — easier to just check the seeds differ.
    assert _stable_seed(42, "foo") != _stable_seed(42, "bar")
    # And, on this fixture, "foo" and "bar" pick different sets:
    assert a.required_node_ids != b.required_node_ids


def test_random_subset_size_capped_to_available():
    """Asking for more than |nodes| just returns all of them."""
    res = asyncio.run(RandomSubsetBaseline(size=999, seed=42).run(_inputs()))
    assert len(res.required_node_ids) == 5  # only 5 nodes in fixture


def test_random_subset_size_zero_returns_empty():
    res = asyncio.run(RandomSubsetBaseline(size=0, seed=42).run(_inputs()))
    assert res.required_node_ids == []
    assert res.nodes == []


def test_random_subset_empty_diagram():
    inputs = ApproachInputs(query="q", diagram={"nodes": [], "edges": []})
    res = asyncio.run(RandomSubsetBaseline(size=3, seed=42).run(inputs))
    assert res.required_node_ids == []


def test_random_subset_negative_size_raises():
    with pytest.raises(ValueError):
        RandomSubsetBaseline(size=-1)


# ---- top_degree -----------------------------------------------------------


def test_top_degree_picks_most_connected():
    """B has degree 3 (A-B, B-C, B-D), A has 2, C has 2, D has 2, E has 1.
    For size=2, B must be in the picked set; the second slot is a tie among
    A/C/D and tie-breaking is lexicographic on node_id, so 'p.A' wins."""
    res = asyncio.run(TopDegreeBaseline(size=2).run(_inputs()))
    assert "p.B" in res.required_node_ids
    assert "p.A" in res.required_node_ids
    assert len(res.required_node_ids) == 2


def test_top_degree_size_capped_to_available():
    res = asyncio.run(TopDegreeBaseline(size=999).run(_inputs()))
    assert len(res.required_node_ids) == 5


def test_top_degree_zero_returns_empty():
    res = asyncio.run(TopDegreeBaseline(size=0).run(_inputs()))
    assert res.required_node_ids == []


def test_top_degree_ignores_self_loops():
    """Self-loops contribute nothing to degree."""
    diagram = _diagram()
    diagram["edges"].append(
        {"node_id_from": "p.E", "node_id_to": "p.E", "description": "Dependency"}
    )
    inputs = ApproachInputs(query="q", diagram=diagram, sample_id="x")
    res = asyncio.run(TopDegreeBaseline(size=1).run(inputs))
    # Without self-loop counting, E has degree 1; B has 3 → B wins.
    assert res.required_node_ids == ["p.B"]


def test_top_degree_negative_size_raises():
    with pytest.raises(ValueError):
        TopDegreeBaseline(size=-1)


# ---- registry integration -------------------------------------------------


def test_baselines_are_registered():
    """The registry exposes the four baselines under their canonical names."""
    from src.approaches import list_approaches

    names = list_approaches()
    for n in ("empty", "full_diagram", "random_subset", "top_degree"):
        assert n in names, f"baseline '{n}' not in registry: {names}"


def test_factory_get_runner_returns_correct_type():
    from src.approaches import get_runner

    assert isinstance(get_runner("empty"), EmptyBaseline)
    assert isinstance(get_runner("full_diagram"), FullDiagramBaseline)
    assert isinstance(get_runner("random_subset"), RandomSubsetBaseline)
    assert isinstance(get_runner("top_degree"), TopDegreeBaseline)


def test_random_subset_reads_size_from_config():
    """Config with a non-default size must be honoured."""
    from src.approaches import get_runner

    # cfg via duck-typed dict-like
    class _Cfg:
        def __init__(self, d):
            self._d = d

        def get(self, k, default=None):
            return self._d.get(k, default)

    cfg = _Cfg({"approaches": _Cfg({"random_subset": _Cfg({"size": 7, "seed": 99})})})
    runner = get_runner("random_subset", cfg)
    assert runner._size == 7  # noqa: SLF001 — testing private attr is fine here
    assert runner._seed == 99


def test_factory_handles_missing_config_gracefully():
    """No config → defaults (size=5, seed=42)."""
    from src.approaches import get_runner

    runner = get_runner("random_subset", None)
    assert runner._size == 5  # noqa: SLF001
