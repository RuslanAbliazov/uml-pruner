"""End-to-end smoke test: run baselines + score on the tiny fixture.

This is the test that catches "renamed a column in the dataset CSV" and
"changed the result file shape" — exactly the kind of breakage that's
silent until you try to actually run the pipeline.

Stays inside the test file system (tmp_path), so it doesn't pollute
data/results/ in the repo.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

FIX_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "tiny"

from src.approaches import get_runner  # noqa: E402
from src.core.io import save_diagram  # noqa: E402
from src.core.types import ApproachInputs  # noqa: E402
from src.eval.annotations import load_dataset  # noqa: E402
from src.eval.evaluator import evaluate_test_set  # noqa: E402
from src.eval.oracle_baselines import predict_central_plus_neighbors, predict_gold_only  # noqa: E402


def _run_approach_on_fixture(approach_name: str, out_dir: Path) -> None:
    """Run a registered approach on the tiny fixture, write per-sample JSONs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_dataset(FIX_DIR / "dataset.csv")
    diagram_path = FIX_DIR / "diagrams_normalized" / "tiny.json"
    import json

    diagram = json.loads(diagram_path.read_text(encoding="utf-8"))

    async def _run():
        runner = get_runner(approach_name)
        for s in samples:
            inputs = ApproachInputs(
                query=s.query, diagram=diagram, sample_id=s.sample_id, repo=s.repo
            )
            result = await runner.run(inputs)
            diagram_out = result.to_diagram()
            save_diagram(diagram_out, out_dir / f"{s.sample_id}.json")
        await runner.aclose()

    asyncio.run(_run())


def test_empty_baseline_e2e(tmp_path: Path):
    out = tmp_path / "empty"
    _run_approach_on_fixture("empty", out)
    result = evaluate_test_set(FIX_DIR / "dataset.csv", str(out))
    # Empty: predicted nothing → all recalls are 0
    assert result.summary["macro"]["mean_recall_overall"] == 0.0
    assert result.summary["macro"]["mean_predicted_size"] == 0.0
    assert result.summary["num_samples"] == 2


def test_full_diagram_baseline_e2e(tmp_path: Path):
    out = tmp_path / "full"
    _run_approach_on_fixture("full_diagram", out)
    result = evaluate_test_set(FIX_DIR / "dataset.csv", str(out))
    # Full diagram → recall = 1.0 (we predicted everything in the gold set).
    assert result.summary["macro"]["mean_recall_overall"] == 1.0
    # Predicted size equals diagram size on every sample (5 nodes).
    assert result.summary["macro"]["mean_predicted_size"] == 5.0


def test_random_subset_baseline_e2e(tmp_path: Path):
    out = tmp_path / "random"
    _run_approach_on_fixture("random_subset", out)
    result = evaluate_test_set(FIX_DIR / "dataset.csv", str(out))
    # Random subset of size 5 with 5-node diagram = the full diagram.
    # Use a different size via direct construction so the test is meaningful:
    assert result.summary["num_samples"] == 2


def test_top_degree_baseline_e2e(tmp_path: Path):
    out = tmp_path / "top_deg"
    _run_approach_on_fixture("top_degree", out)
    result = evaluate_test_set(FIX_DIR / "dataset.csv", str(out))
    # Should produce 5-or-fewer predicted nodes on each sample (size default 5).
    assert all(m.predicted_count <= 5 for m in result.per_sample)


def test_oracle_gold_only_is_perfect(tmp_path: Path):
    """Oracle gold_only must give F1 = 1.0 on every sample."""
    import json

    out = tmp_path / "gold_only"
    out.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(FIX_DIR / "dataset.csv")
    diagram = json.loads(
        (FIX_DIR / "diagrams_normalized" / "tiny.json").read_text(encoding="utf-8")
    )
    for s in samples:
        keep = predict_gold_only(s, diagram)
        nodes_out = [n for n in diagram["nodes"] if n["node_id"] in keep]
        edges_out = [
            e for e in diagram["edges"]
            if e["node_id_from"] in keep and e["node_id_to"] in keep
        ]
        save_diagram(
            {
                "nodes": nodes_out,
                "edges": edges_out,
                "metadata": {"approach": "oracle_gold_only", "is_oracle": True},
            },
            out / f"{s.sample_id}.json",
        )

    result = evaluate_test_set(FIX_DIR / "dataset.csv", str(out))
    # Both samples should give F1 = 1.0 since predicted == gold exactly.
    for m in result.per_sample:
        assert m.recall_overall == 1.0
        # precision_known = 1.0 (no annotated irrelevants in fixture, so no FPs)
        assert m.precision_known == 1.0


def test_oracle_central_plus_neighbors_e2e(tmp_path: Path):
    """Oracle central_plus_neighbors writes valid output the evaluator reads."""
    import json

    out = tmp_path / "cpn"
    out.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(FIX_DIR / "dataset.csv")
    diagram = json.loads(
        (FIX_DIR / "diagrams_normalized" / "tiny.json").read_text(encoding="utf-8")
    )
    for s in samples:
        keep = predict_central_plus_neighbors(s, diagram)
        save_diagram(
            {
                "nodes": [n for n in diagram["nodes"] if n["node_id"] in keep],
                "edges": [
                    e for e in diagram["edges"]
                    if e["node_id_from"] in keep and e["node_id_to"] in keep
                ],
                "metadata": {"approach": "oracle_central_plus_neighbors", "is_oracle": True},
            },
            out / f"{s.sample_id}.json",
        )

    result = evaluate_test_set(FIX_DIR / "dataset.csv", str(out))
    # On the tiny fixture, central=p.A → {A, B, C} which equals gold for sample_one.
    # central=p.D → {D, B, E} which contains both required (D) and useful (E) plus B.
    # Recall_overall on both samples should be 1.0.
    assert result.summary["macro"]["mean_recall_overall"] == 1.0
