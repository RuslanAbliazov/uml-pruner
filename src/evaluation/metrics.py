"""Core evaluation metrics for UML pruning.

Ground truth format (from annotations.csv, 'entity_annotations' column):
    {"full.class.Name": "required" | "useful" | "irrelevant"}

A sample is evaluated against a predicted set of node_ids output by the
pipeline. We compute per-sample precision / recall / F1 and support
aggregation across a test set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Label = str  # "required" | "useful" | "irrelevant"


@dataclass
class SampleMetrics:
    """Metrics for a single (query, diagram) sample."""

    sample_id: str
    query: str

    # Counts
    predicted_count: int
    required_total: int
    useful_total: int
    irrelevant_total: int

    tp_required: int
    tp_useful: int
    fp_irrelevant: int  # predicted ∩ irrelevant (known false positives)
    fn_required: int
    fn_useful: int

    # Rates
    recall_required: float
    recall_useful: float
    recall_overall: float
    precision_strict: float  # relevant / predicted (among known labels)
    precision_known: float  # 1 - fp_irrelevant / predicted
    f1_score: float

    # Diagnostics
    missed_required: list[str] = field(default_factory=list)
    missed_useful: list[str] = field(default_factory=list)
    false_positives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "query": self.query,
            "predicted_count": self.predicted_count,
            "required_total": self.required_total,
            "useful_total": self.useful_total,
            "irrelevant_total": self.irrelevant_total,
            "tp_required": self.tp_required,
            "tp_useful": self.tp_useful,
            "fp_irrelevant": self.fp_irrelevant,
            "fn_required": self.fn_required,
            "fn_useful": self.fn_useful,
            "recall_required": self.recall_required,
            "recall_useful": self.recall_useful,
            "recall_overall": self.recall_overall,
            "precision_strict": self.precision_strict,
            "precision_known": self.precision_known,
            "f1_score": self.f1_score,
            "missed_required": self.missed_required,
            "missed_useful": self.missed_useful,
            "false_positives": self.false_positives,
        }


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_sample(
    sample_id: str,
    query: str,
    predicted_ids: set[str] | list[str],
    annotations: dict[str, Label],
) -> SampleMetrics:
    """Evaluate a single sample.

    Args:
        sample_id: Identifier for the sample.
        query: User query string.
        predicted_ids: node_ids produced by the pipeline.
        annotations: Ground-truth labels: class_name -> "required"|"useful"|"irrelevant".

    Returns:
        SampleMetrics with all counts and rates.
    """
    predicted = set(predicted_ids)

    required = {cls for cls, lbl in annotations.items() if lbl == "required"}
    useful = {cls for cls, lbl in annotations.items() if lbl == "useful"}
    irrelevant = {cls for cls, lbl in annotations.items() if lbl == "irrelevant"}
    relevant = required | useful

    tp_required_set = predicted & required
    tp_useful_set = predicted & useful
    fp_set = predicted & irrelevant  # known-bad predictions
    fn_required_set = required - predicted
    fn_useful_set = useful - predicted

    # Recall metrics
    recall_required = _safe_div(len(tp_required_set), len(required))
    recall_useful = _safe_div(len(tp_useful_set), len(useful))
    recall_overall = _safe_div(len(tp_required_set) + len(tp_useful_set), len(relevant))

    # Precision:
    # - precision_strict: fraction of predictions that are KNOWN relevant.
    #   Harsh: unannotated predictions count against us.
    # - precision_known: 1 - (known-bad / predicted). Lenient: only known
    #   irrelevant predictions count against us.
    precision_strict = _safe_div(
        len(tp_required_set) + len(tp_useful_set), len(predicted)
    )
    precision_known = 1.0 - _safe_div(len(fp_set), len(predicted)) if predicted else 0.0

    # F1 uses precision_known (consistent with practice of "don't penalise
    # unlabeled predictions"), combined with recall_overall.
    f1 = (
        2 * precision_known * recall_overall / (precision_known + recall_overall)
        if (precision_known + recall_overall) > 0
        else 0.0
    )

    return SampleMetrics(
        sample_id=sample_id,
        query=query,
        predicted_count=len(predicted),
        required_total=len(required),
        useful_total=len(useful),
        irrelevant_total=len(irrelevant),
        tp_required=len(tp_required_set),
        tp_useful=len(tp_useful_set),
        fp_irrelevant=len(fp_set),
        fn_required=len(fn_required_set),
        fn_useful=len(fn_useful_set),
        recall_required=recall_required,
        recall_useful=recall_useful,
        recall_overall=recall_overall,
        precision_strict=precision_strict,
        precision_known=precision_known,
        f1_score=f1,
        missed_required=sorted(fn_required_set),
        missed_useful=sorted(fn_useful_set),
        false_positives=sorted(fp_set),
    )


def aggregate_metrics(samples: list[SampleMetrics]) -> dict[str, Any]:
    """Aggregate metrics across samples.

    Reports mean/median/p90 for each rate, plus totals and worst cases.
    """
    if not samples:
        return {"num_samples": 0}

    def mean(vals: list[float]) -> float:
        return sum(vals) / len(vals)

    def percentile(vals: list[float], p: float) -> float:
        s = sorted(vals)
        if not s:
            return 0.0
        idx = min(int(p * len(s)), len(s) - 1)
        return s[idx]

    f1s = [m.f1_score for m in samples]
    recalls_req = [m.recall_required for m in samples]
    recalls_use = [m.recall_useful for m in samples]
    recalls_all = [m.recall_overall for m in samples]
    precisions_k = [m.precision_known for m in samples]
    precisions_s = [m.precision_strict for m in samples]
    sizes = [m.predicted_count for m in samples]

    # Totals (micro-averaged view: combine counts first)
    total_tp_req = sum(m.tp_required for m in samples)
    total_tp_use = sum(m.tp_useful for m in samples)
    total_fp = sum(m.fp_irrelevant for m in samples)
    total_req = sum(m.required_total for m in samples)
    total_use = sum(m.useful_total for m in samples)
    total_pred = sum(m.predicted_count for m in samples)

    micro_recall_required = _safe_div(total_tp_req, total_req)
    micro_recall_useful = _safe_div(total_tp_use, total_use)
    micro_recall_overall = _safe_div(total_tp_req + total_tp_use, total_req + total_use)
    micro_precision_known = 1.0 - _safe_div(total_fp, total_pred) if total_pred else 0.0
    micro_f1 = (
        2
        * micro_precision_known
        * micro_recall_overall
        / (micro_precision_known + micro_recall_overall)
        if (micro_precision_known + micro_recall_overall) > 0
        else 0.0
    )

    worst = sorted(samples, key=lambda m: m.f1_score)[:5]
    best = sorted(samples, key=lambda m: m.f1_score, reverse=True)[:3]

    return {
        "num_samples": len(samples),
        "macro": {
            "mean_recall_required": mean(recalls_req),
            "mean_recall_useful": mean(recalls_use),
            "mean_recall_overall": mean(recalls_all),
            "mean_precision_known": mean(precisions_k),
            "mean_precision_strict": mean(precisions_s),
            "mean_f1_score": mean(f1s),
            "median_f1": percentile(f1s, 0.5),
            "p90_f1": percentile(f1s, 0.9),
            "p10_f1": percentile(f1s, 0.1),
            "mean_predicted_size": mean(sizes),
        },
        "micro": {
            "recall_required": micro_recall_required,
            "recall_useful": micro_recall_useful,
            "recall_overall": micro_recall_overall,
            "precision_known": micro_precision_known,
            "f1_score": micro_f1,
            "total_required": total_req,
            "total_useful": total_use,
            "total_predicted": total_pred,
            "total_tp_required": total_tp_req,
            "total_tp_useful": total_tp_use,
            "total_fp_irrelevant": total_fp,
        },
        "worst_samples": [
            {
                "sample_id": m.sample_id,
                "f1": m.f1_score,
                "recall_required": m.recall_required,
                "precision_known": m.precision_known,
                "query": m.query[:120],
            }
            for m in worst
        ],
        "best_samples": [
            {
                "sample_id": m.sample_id,
                "f1": m.f1_score,
                "recall_required": m.recall_required,
                "query": m.query[:120],
            }
            for m in best
        ],
    }
