"""Inter-annotator agreement (IAA).

This module computes IAA on RAW annotation rows from ``annotations.csv``,
NOT on the consolidated ``data/dataset.csv`` (where votes have already been
merged and ``irrelevant`` labels dropped — the multi-annotator signal needed
for IAA is gone).

What we report
--------------

* ``percent_agreement``   — pairwise fraction of nodes both annotators
                            labelled the same. Easy to read, ignores
                            chance.
* ``cohens_kappa``        — pairwise, chance-corrected agreement on three
                            labels (required / useful / irrelevant). Range
                            [-1, 1]; > 0.6 substantial; > 0.8 near-perfect.
* ``fleiss_kappa``        — multi-rater extension; useful when more than 2
                            annotators saw the same sample.

Universe of nodes
-----------------

For each (sample, pair-of-annotators), the universe of nodes is the
**union** of node_ids labelled by either annotator. A node mentioned by one
annotator but not the other is treated as ``"irrelevant"`` for the
non-mentioning annotator (the implicit "I saw the node and decided not to
include it" interpretation, which matches how the labelling UI works in
this project: annotators see a fixed candidate set and tick boxes).

If you'd rather restrict to the intersection of labelled nodes, pass
``policy="intersection"`` — it's mostly useful when annotators worked on
genuinely different node sets (e.g. different threshold sliders). Default
is ``"union_with_implicit_irrelevant"``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Literal

LABELS = ("required", "useful", "irrelevant")
BINARY_LABELS = ("required_or_useful", "irrelevant")
Label = str  # one of LABELS

UniversePolicy = Literal["union_with_implicit_irrelevant", "intersection"]


# ----------------------------------------------------------------------------
# Raw annotation loading
# ----------------------------------------------------------------------------


@dataclass
class RawAnnotation:
    """One row from ``annotations.csv``."""

    sample_id: str
    annotator: str
    status: str
    labels: dict[str, Label]


def _parse_labels_field(raw: str) -> dict[str, Label]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Tolerate the doubled-quote dialect occasionally produced by Excel.
        return json.loads(raw.replace('""', '"'))


def load_raw_annotations(
    csv_path: str | Path,
    *,
    finalized_only: bool = True,
    exclude_annotators: Iterable[str] = (),
) -> list[RawAnnotation]:
    """Read ``annotations.csv`` and return every raw row that has labels.

    Args:
        csv_path: Path to ``annotations.csv`` (raw, NOT the consolidated
            dataset).
        finalized_only: If True, drop rows whose status != "Finalized".
        exclude_annotators: Annotator names to ignore entirely. Matches the
            ``--exclude-annotator`` flag of ``scripts/build_dataset.py``.

    Returns:
        List of :class:`RawAnnotation`. Rows with empty ``labels`` are
        dropped (they carry no IAA signal).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"raw annotations CSV not found: {path}")

    excluded = {a.strip() for a in exclude_annotators if a and a.strip()}
    out: list[RawAnnotation] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = (row.get("status") or "").strip()
            if finalized_only and status != "Finalized":
                continue
            annotator = (row.get("annotator") or "").strip()
            if not annotator or annotator in excluded:
                continue
            labels = _parse_labels_field(row.get("entity_annotations") or "")
            if not labels:
                continue
            out.append(
                RawAnnotation(
                    sample_id=row.get("sample_id", "").strip(),
                    annotator=annotator,
                    status=status,
                    labels={k: v for k, v in labels.items() if v in LABELS},
                )
            )
    return out


def group_by_sample(
    rows: list[RawAnnotation],
) -> dict[str, list[RawAnnotation]]:
    """Group raw rows by sample_id."""
    out: dict[str, list[RawAnnotation]] = {}
    for r in rows:
        out.setdefault(r.sample_id, []).append(r)
    return out


# ----------------------------------------------------------------------------
# Per-pair / per-sample agreement primitives
# ----------------------------------------------------------------------------


def _label_to_binary(label: str) -> str:
    """Merge required/useful into a single positive class."""
    if label in ("required", "useful"):
        return "required_or_useful"
    return label


def _aligned_labels(
    a: RawAnnotation,
    b: RawAnnotation,
    policy: UniversePolicy,
    merge_required_useful: bool = False,
) -> tuple[list[Label], list[Label]]:
    """Build two parallel label arrays for the two annotators.

    Universe selection follows ``policy``. For the default
    ``"union_with_implicit_irrelevant"``, missing labels are filled with
    "irrelevant" — i.e. "annotator saw it and didn't tick it".
    """
    if policy == "intersection":
        ids = sorted(set(a.labels) & set(b.labels))
        la = [a.labels[i] for i in ids]
        lb = [b.labels[i] for i in ids]
    else:
        # default: union with implicit irrelevant
        ids = sorted(set(a.labels) | set(b.labels))
        la = [a.labels.get(i, "irrelevant") for i in ids]
        lb = [b.labels.get(i, "irrelevant") for i in ids]

    if merge_required_useful:
        la = [_label_to_binary(l) for l in la]
        lb = [_label_to_binary(l) for l in lb]

    return la, lb


def percent_agreement(la: list[Label], lb: list[Label]) -> float:
    """Fraction of positions on which the two annotators agree."""
    if not la:
        return float("nan")
    n = len(la)
    same = sum(1 for x, y in zip(la, lb) if x == y)
    return same / n


def cohens_kappa(
    la: list[Label],
    lb: list[Label],
    labels: tuple[str, ...] = LABELS,
) -> float:
    """Cohen's κ for two annotators on a fixed label set.

    Returns ``nan`` if the input is empty. Returns ``1.0`` if all labels
    agree and ``pe == 1.0`` (degenerate single-label case) — by convention
    here, perfect agreement on one label is not penalised.
    """
    n = len(la)
    if n == 0 or n != len(lb):
        return float("nan")

    # Observed agreement
    po = sum(1 for x, y in zip(la, lb) if x == y) / n

    # Marginal distributions
    pa = {lab: 0 for lab in labels}
    pb = {lab: 0 for lab in labels}
    for x in la:
        pa[x] = pa.get(x, 0) + 1
    for y in lb:
        pb[y] = pb.get(y, 0) + 1

    # Chance agreement
    pe = sum((pa[lab] / n) * (pb[lab] / n) for lab in labels)

    if pe == 1.0:
        # Both annotators put everything in the same single class. By
        # convention return 1.0 — perfect (degenerate) agreement.
        return 1.0
    return (po - pe) / (1.0 - pe)


def fleiss_kappa(
    annotations: list[RawAnnotation],
    *,
    policy: UniversePolicy = "union_with_implicit_irrelevant",
    labels: tuple[str, ...] = LABELS,
    merge_required_useful: bool = False,
) -> float:
    """Fleiss' κ across N ≥ 2 annotators on the same sample.

    Standard definition (Fleiss 1971): for each item we have N annotator
    votes spread over k categories; agreement on the item is the average
    pairwise agreement among the N annotators. κ corrects for chance.

    Returns ``nan`` when fewer than 2 annotators are supplied or when there
    is nothing to score.
    """
    n_raters = len(annotations)
    if n_raters < 2:
        return float("nan")

    if merge_required_useful:
        annotations = [
            RawAnnotation(
                sample_id=a.sample_id,
                annotator=a.annotator,
                status=a.status,
                labels={
                    nid: _label_to_binary(lab) for nid, lab in a.labels.items()
                },
            )
            for a in annotations
        ]
        labels = BINARY_LABELS

    if policy == "intersection":
        ids = set(annotations[0].labels)
        for a in annotations[1:]:
            ids &= set(a.labels)
        ids = sorted(ids)

        def label_of(a: RawAnnotation, nid: str) -> str | None:
            return a.labels.get(nid)
    else:
        ids = set()
        for a in annotations:
            ids |= set(a.labels)
        ids = sorted(ids)

        def label_of(a: RawAnnotation, nid: str) -> str | None:
            return a.labels.get(nid, "irrelevant")

    if not ids:
        return float("nan")

    # Build the n_items × n_categories matrix of vote counts.
    cat_index = {lab: i for i, lab in enumerate(labels)}
    n_cats = len(labels)
    n_items = len(ids)

    counts = [[0] * n_cats for _ in range(n_items)]
    valid_items = 0
    p_j = [0] * n_cats  # column totals (summed across all items)
    items_used = 0

    for i, nid in enumerate(ids):
        votes_here = 0
        for a in annotations:
            lab = label_of(a, nid)
            if lab is None or lab not in cat_index:
                continue
            counts[i][cat_index[lab]] += 1
            votes_here += 1
        if votes_here == n_raters:
            valid_items += 1
            for j in range(n_cats):
                p_j[j] += counts[i][j]
        # if votes_here != n_raters under "intersection" we skip the item

    if valid_items == 0:
        return float("nan")

    total_votes = valid_items * n_raters
    p_j = [c / total_votes for c in p_j]
    pe = sum(p ** 2 for p in p_j)

    # Per-item agreement P_i.
    pi_sum = 0.0
    used = 0
    for i in range(n_items):
        row_sum = sum(counts[i])
        if row_sum != n_raters:
            continue
        pi = (sum(c * c for c in counts[i]) - n_raters) / (n_raters * (n_raters - 1))
        pi_sum += pi
        used += 1
    p_bar = pi_sum / used

    if pe >= 1.0:
        return 1.0
    return (p_bar - pe) / (1.0 - pe)


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------


@dataclass
class SampleIAA:
    sample_id: str
    n_annotators: int
    annotators: list[str]
    n_nodes_universe: int
    pairwise_kappas: list[dict[str, Any]]  # [{"pair": [a, b], "kappa": ..., "agreement": ...}]
    fleiss_kappa: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "n_annotators": self.n_annotators,
            "annotators": self.annotators,
            "n_nodes_universe": self.n_nodes_universe,
            "pairwise_kappas": self.pairwise_kappas,
            "fleiss_kappa": self.fleiss_kappa,
        }


@dataclass
class IAAReport:
    per_sample: list[SampleIAA] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "per_sample": [s.to_dict() for s in self.per_sample],
        }


def _safe_mean(xs: list[float]) -> float:
    """Mean ignoring NaNs. Returns nan if every element is nan."""
    clean = [x for x in xs if x == x]  # NaN != NaN
    return sum(clean) / len(clean) if clean else float("nan")


def compute_iaa(
    rows: list[RawAnnotation],
    *,
    policy: UniversePolicy = "union_with_implicit_irrelevant",
    min_annotators: int = 2,
    merge_required_useful: bool = False,
) -> IAAReport:
    """Compute per-sample and summary IAA.

    Samples with fewer than ``min_annotators`` are silently skipped — there's
    no IAA to compute.
    """
    by_sample = group_by_sample(rows)
    per_sample: list[SampleIAA] = []

    for sid, anns in sorted(by_sample.items()):
        if len(anns) < min_annotators:
            continue

        anns_sorted = sorted(anns, key=lambda r: r.annotator)

        # universe size for diagnostics
        if policy == "intersection":
            universe = set(anns_sorted[0].labels)
            for a in anns_sorted[1:]:
                universe &= set(a.labels)
        else:
            universe = set()
            for a in anns_sorted:
                universe |= set(a.labels)

        pairwise: list[dict[str, Any]] = []
        active_labels = BINARY_LABELS if merge_required_useful else LABELS

        for a, b in combinations(anns_sorted, 2):
            la, lb = _aligned_labels(a, b, policy,
                                     merge_required_useful=merge_required_useful)
            pairwise.append(
                {
                    "pair": [a.annotator, b.annotator],
                    "kappa": cohens_kappa(la, lb, labels=active_labels),
                    "agreement": percent_agreement(la, lb),
                    "n_compared": len(la),
                }
            )

        fk = fleiss_kappa(
            anns_sorted,
            policy=policy,
            labels=active_labels,
            merge_required_useful=merge_required_useful,
        ) if len(anns_sorted) >= 2 else float("nan")

        per_sample.append(
            SampleIAA(
                sample_id=sid,
                n_annotators=len(anns_sorted),
                annotators=[a.annotator for a in anns_sorted],
                n_nodes_universe=len(universe),
                pairwise_kappas=pairwise,
                fleiss_kappa=fk,
            )
        )

    # Aggregates
    all_kappas = [p["kappa"] for s in per_sample for p in s.pairwise_kappas]
    all_agreements = [p["agreement"] for s in per_sample for p in s.pairwise_kappas]
    fleiss_vals = [s.fleiss_kappa for s in per_sample]

    summary = {
        "n_samples_with_iaa": len(per_sample),
        "n_pairwise_comparisons": len(all_kappas),
        "mean_cohens_kappa": _safe_mean(all_kappas),
        "mean_percent_agreement": _safe_mean(all_agreements),
        "mean_fleiss_kappa": _safe_mean(fleiss_vals),
        "policy": policy,
        "binary_merge": merge_required_useful,
    }

    return IAAReport(per_sample=per_sample, summary=summary)


def format_summary(report: IAAReport) -> str:
    """Plain-text human-readable summary."""
    s = report.summary
    if s.get("n_samples_with_iaa", 0) == 0:
        return "No multi-annotator samples found — IAA cannot be computed."

    def fmt(x: float) -> str:
        return f"{x:.3f}" if x == x else "nan"  # NaN check

    lines = [
        "=" * 60,
        "INTER-ANNOTATOR AGREEMENT",
        "=" * 60,
        f"  Samples with >= 2 annotators: {s['n_samples_with_iaa']}",
        f"  Pairwise comparisons:         {s['n_pairwise_comparisons']}",
        f"  Universe policy:              {s['policy']}",
        "",
        f"  Mean Cohen's κ:        {fmt(s['mean_cohens_kappa'])}",
        f"  Mean percent agreement:{fmt(s['mean_percent_agreement'])}",
        f"  Mean Fleiss' κ:        {fmt(s['mean_fleiss_kappa'])}",
        "=" * 60,
    ]
    return "\n".join(lines)