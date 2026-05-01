"""Unit tests for evaluation metrics."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import aggregate_metrics, evaluate_sample


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def test_perfect_prediction():
    """All required+useful predicted, no irrelevant. F1 = 1.0."""
    annotations = {
        "A": "required",
        "B": "required",
        "C": "useful",
        "D": "irrelevant",
    }
    predicted = {"A", "B", "C"}
    m = evaluate_sample("s1", "q1", predicted, annotations)

    assert m.tp_required == 2
    assert m.tp_useful == 1
    assert m.fp_irrelevant == 0
    assert approx(m.recall_required, 1.0)
    assert approx(m.recall_useful, 1.0)
    assert approx(m.recall_overall, 1.0)
    assert approx(m.precision_known, 1.0)
    assert approx(m.f1_score, 1.0)


def test_missed_required():
    """Missing a required class lowers recall_required."""
    annotations = {"A": "required", "B": "required", "C": "useful"}
    predicted = {"A", "C"}
    m = evaluate_sample("s", "q", predicted, annotations)

    assert m.tp_required == 1
    assert m.fn_required == 1
    assert approx(m.recall_required, 0.5)
    assert approx(m.recall_overall, 2 / 3)
    assert "B" in m.missed_required


def test_false_positive_irrelevant():
    """Predicting an irrelevant class lowers precision_known."""
    annotations = {"A": "required", "B": "irrelevant"}
    predicted = {"A", "B"}
    m = evaluate_sample("s", "q", predicted, annotations)

    assert m.tp_required == 1
    assert m.fp_irrelevant == 1
    assert approx(m.recall_required, 1.0)
    # precision_known = 1 - 1/2 = 0.5
    assert approx(m.precision_known, 0.5)
    # precision_strict = relevant(1) / predicted(2) = 0.5
    assert approx(m.precision_strict, 0.5)


def test_empty_prediction():
    annotations = {"A": "required"}
    predicted: set[str] = set()
    m = evaluate_sample("s", "q", predicted, annotations)
    assert m.predicted_count == 0
    assert approx(m.recall_required, 0.0)
    assert approx(m.precision_known, 0.0)
    assert approx(m.f1_score, 0.0)


def test_no_annotations():
    """No ground truth -> recall = 0 by convention (safe divisions)."""
    predicted = {"A"}
    m = evaluate_sample("s", "q", predicted, {})
    assert m.required_total == 0
    assert approx(m.recall_required, 0.0)
    assert approx(m.recall_useful, 0.0)
    # precision_known = 1.0 since no known FP
    assert approx(m.precision_known, 1.0)


def test_aggregate_macro():
    anns = {"A": "required", "B": "useful"}
    m1 = evaluate_sample("s1", "q", {"A", "B"}, anns)  # f1=1.0
    m2 = evaluate_sample("s2", "q", {"A"}, anns)  # recall_overall=0.5
    summary = aggregate_metrics([m1, m2])
    assert summary["num_samples"] == 2
    assert approx(summary["macro"]["mean_f1_score"], (m1.f1_score + m2.f1_score) / 2)
    assert "micro" in summary
    assert summary["micro"]["total_required"] == 2
    assert summary["micro"]["total_useful"] == 2


def run_all():
    tests = [
        test_perfect_prediction,
        test_missed_required,
        test_false_positive_irrelevant,
        test_empty_prediction,
        test_no_annotations,
        test_aggregate_macro,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")


if __name__ == "__main__":
    run_all()
