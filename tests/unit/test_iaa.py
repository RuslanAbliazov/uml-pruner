"""Unit tests for inter-annotator agreement.

Hand-computed kappa values are verified against the implementation. We
check the canonical edge cases: perfect agreement, total disagreement,
single-label degenerate case, mixed labels with a known κ.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.iaa import (  # noqa: E402
    LABELS,
    RawAnnotation,
    cohens_kappa,
    compute_iaa,
    fleiss_kappa,
    load_raw_annotations,
    percent_agreement,
)


FIX_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "tiny"


# ---- low-level metrics ----------------------------------------------------


def test_percent_agreement_perfect():
    la = ["required", "useful", "irrelevant"]
    lb = list(la)
    assert percent_agreement(la, lb) == 1.0


def test_percent_agreement_half():
    la = ["required", "required", "useful", "useful"]
    lb = ["required", "useful", "useful", "irrelevant"]
    assert percent_agreement(la, lb) == 0.5


def test_cohens_kappa_perfect_agreement():
    la = ["required", "useful", "irrelevant", "required"]
    assert cohens_kappa(la, la) == pytest.approx(1.0)


def test_cohens_kappa_complete_disagreement_with_balanced_marginals():
    """Marginals: A=[1,1,1] B=[1,1,1] (one of each label per annotator).
    po = 0; pe = 1/9 + 1/9 + 1/9 = 1/3.
    κ = (0 - 1/3) / (1 - 1/3) = -0.5.
    """
    la = ["required", "useful", "irrelevant"]
    lb = ["useful", "irrelevant", "required"]
    assert cohens_kappa(la, lb) == pytest.approx(-0.5, rel=1e-6)


def test_cohens_kappa_known_case():
    """Hand-calculated reference.

    Items: 4
        A: [required, required, useful, irrelevant]
        B: [required, useful,   useful, irrelevant]

    po = 3/4 = 0.75
    A marginals: required=2/4, useful=1/4, irrelevant=1/4
    B marginals: required=1/4, useful=2/4, irrelevant=1/4
    pe = (2/4)(1/4) + (1/4)(2/4) + (1/4)(1/4) = 2/16 + 2/16 + 1/16 = 5/16 = 0.3125
    κ = (0.75 - 0.3125) / (1 - 0.3125) = 0.4375 / 0.6875 ≈ 0.636363...
    """
    la = ["required", "required", "useful", "irrelevant"]
    lb = ["required", "useful", "useful", "irrelevant"]
    assert cohens_kappa(la, lb) == pytest.approx(7 / 11, rel=1e-6)


def test_cohens_kappa_degenerate_single_label():
    """Both annotators put everything in the same single label.

    pe = 1.0 → κ formally undefined; we return 1.0 by convention.
    """
    la = ["required"] * 5
    lb = ["required"] * 5
    assert cohens_kappa(la, lb) == 1.0


def test_cohens_kappa_empty_input_is_nan():
    assert math.isnan(cohens_kappa([], []))


# ---- Fleiss' kappa --------------------------------------------------------


def test_fleiss_kappa_perfect_agreement():
    """3 annotators, all label everything the same → κ = 1.0."""
    a = RawAnnotation("s", "A", "Finalized", {"n1": "required", "n2": "useful"})
    b = RawAnnotation("s", "B", "Finalized", {"n1": "required", "n2": "useful"})
    c = RawAnnotation("s", "C", "Finalized", {"n1": "required", "n2": "useful"})
    assert fleiss_kappa([a, b, c]) == pytest.approx(1.0)


def test_fleiss_kappa_too_few_annotators_is_nan():
    a = RawAnnotation("s", "A", "Finalized", {"n1": "required"})
    assert math.isnan(fleiss_kappa([a]))


def test_fleiss_kappa_three_annotators_mixed():
    """Sanity check that we get a finite, sensible κ on a mixed example."""
    a = RawAnnotation("s", "A", "Finalized",
                      {"n1": "required", "n2": "useful", "n3": "irrelevant"})
    b = RawAnnotation("s", "B", "Finalized",
                      {"n1": "required", "n2": "useful", "n3": "irrelevant"})
    c = RawAnnotation("s", "C", "Finalized",
                      {"n1": "required", "n2": "irrelevant", "n3": "irrelevant"})
    fk = fleiss_kappa([a, b, c])
    assert not math.isnan(fk)
    assert 0.0 < fk < 1.0


# ---- end-to-end IAA report ------------------------------------------------


def test_compute_iaa_on_fixture():
    """Full IAA pipeline on the tiny fixture.

    Fixture has:
      * s1 — Alice / Bob / Carol — 3 annotators, mixed labels
      * s2 — Alice / Bob — 2 annotators, perfect agreement → κ = 1.0
      * s3 — Alice only — 1 annotator, skipped
      * s4 — Alice, status="In Progress" — dropped by default
    """
    rows = load_raw_annotations(FIX_DIR / "annotations_raw.csv")
    # 6 finalized rows (s4 is "In Progress" → dropped)
    assert len(rows) == 6

    report = compute_iaa(rows)
    sample_ids = {s.sample_id for s in report.per_sample}
    # s3 has only 1 annotator → skipped; s4 in-progress → dropped earlier.
    assert sample_ids == {"s1", "s2"}

    s2 = next(s for s in report.per_sample if s.sample_id == "s2")
    assert s2.n_annotators == 2
    assert len(s2.pairwise_kappas) == 1
    assert s2.pairwise_kappas[0]["kappa"] == pytest.approx(1.0)
    assert s2.pairwise_kappas[0]["agreement"] == pytest.approx(1.0)

    s1 = next(s for s in report.per_sample if s.sample_id == "s1")
    # 3 annotators → 3 pairwise comparisons
    assert s1.n_annotators == 3
    assert len(s1.pairwise_kappas) == 3
    # All kappas finite, in [-1, 1]
    for p in s1.pairwise_kappas:
        assert -1.0 <= p["kappa"] <= 1.0


def test_compute_iaa_excludes_annotator():
    rows_all = load_raw_annotations(FIX_DIR / "annotations_raw.csv")
    rows_no_carol = load_raw_annotations(
        FIX_DIR / "annotations_raw.csv",
        exclude_annotators=["Carol"],
    )
    assert any(r.annotator == "Carol" for r in rows_all)
    assert all(r.annotator != "Carol" for r in rows_no_carol)


def test_compute_iaa_intersection_policy():
    """Intersection policy compares only nodes BOTH annotators labelled."""
    a = RawAnnotation("s", "A", "Finalized",
                      {"n1": "required", "n2": "useful"})
    b = RawAnnotation("s", "B", "Finalized",
                      {"n1": "required", "n3": "useful"})
    report = compute_iaa([a, b], policy="intersection")
    # The intersection is {n1}, one item, both labelled "required" → agreement = 1.0
    pair = report.per_sample[0].pairwise_kappas[0]
    assert pair["n_compared"] == 1
    assert pair["agreement"] == pytest.approx(1.0)


def test_compute_iaa_summary_handles_empty_input():
    report = compute_iaa([])
    assert report.summary["n_samples_with_iaa"] == 0
    assert math.isnan(report.summary["mean_cohens_kappa"])


# ---- raw loader -----------------------------------------------------------


def test_load_raw_annotations_filters_non_finalized():
    rows = load_raw_annotations(FIX_DIR / "annotations_raw.csv")
    statuses = {r.status for r in rows}
    assert statuses == {"Finalized"}


def test_load_raw_annotations_can_include_in_progress():
    rows = load_raw_annotations(
        FIX_DIR / "annotations_raw.csv", finalized_only=False
    )
    statuses = {r.status for r in rows}
    assert "In Progress" in statuses


def test_load_raw_annotations_drops_unknown_labels():
    """A row with garbage labels must not blow up the loader."""
    # We just verify the loader keeps only labels in the LABELS tuple.
    rows = load_raw_annotations(FIX_DIR / "annotations_raw.csv")
    for r in rows:
        for lab in r.labels.values():
            assert lab in LABELS
