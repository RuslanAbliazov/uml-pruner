"""High-level evaluator: read annotations, pair with pipeline results, score."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.eval.annotations import AnnotationSample, load_dataset
from src.eval.metrics import (
    SampleMetrics,
    aggregate_metrics,
    evaluate_sample,
)
from src.core.io import load_diagram
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    summary: dict[str, Any]
    per_sample: list[SampleMetrics]
    missing_results: list[str]  # sample_ids with no predicted file

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "per_sample": [m.to_dict() for m in self.per_sample],
            "missing_results": self.missing_results,
        }


def _predicted_node_ids(result_path: Path) -> set[str]:
    """Extract predicted node_ids from a pipeline output JSON."""
    data = load_diagram(result_path)
    return {n["node_id"] for n in data.get("nodes", [])}


def evaluate_test_set(
    dataset_csv: str | Path,
    results_dir: str | Path,
    result_filename_template: str = "{sample_id}.json",
) -> EvaluationResult:
    """Evaluate every sample in the consolidated dataset.

    Args:
        dataset_csv: Path to the dataset CSV produced by
            `scripts/build_dataset.py`.
        results_dir: Directory containing pipeline result JSON files.
        result_filename_template: How result files are named. Uses {sample_id}.

    Returns:
        EvaluationResult with aggregate summary and per-sample metrics.
    """
    results_dir = Path(results_dir)
    samples = load_dataset(dataset_csv)
    logger.info("Loaded %d samples from dataset", len(samples))

    per_sample: list[SampleMetrics] = []
    missing: list[str] = []

    for s in samples:
        result_file = results_dir / result_filename_template.format(
            sample_id=s.sample_id
        )
        if not result_file.exists():
            logger.warning(
                "Missing result for sample %s (%s)", s.sample_id, result_file
            )
            missing.append(s.sample_id)
            continue

        try:
            predicted = _predicted_node_ids(result_file)
        except Exception:
            logger.exception("Failed to read %s", result_file)
            missing.append(s.sample_id)
            continue

        metrics = evaluate_sample(
            sample_id=s.sample_id,
            query=s.query,
            predicted_ids=predicted,
            annotations=s.annotations,
        )
        per_sample.append(metrics)

    summary = aggregate_metrics(per_sample)
    summary["evaluated_samples"] = len(per_sample)
    summary["missing_samples"] = len(missing)

    return EvaluationResult(
        summary=summary, per_sample=per_sample, missing_results=missing
    )


def format_summary_report(result: EvaluationResult) -> str:
    """Human-readable text summary."""
    s = result.summary
    if s.get("num_samples", 0) == 0:
        return f"No samples evaluated. Missing: {len(result.missing_results)}"

    macro = s["macro"]
    micro = s["micro"]

    lines = [
        "=" * 70,
        f"UML PRUNER EVALUATION REPORT",
        "=" * 70,
        f"Evaluated samples: {s['num_samples']}",
        f"Missing results:   {s.get('missing_samples', 0)}",
        "",
        "--- MACRO METRICS (average across samples) ---",
        f"  Recall (required):    {macro['mean_recall_required']:.3f}",
        f"  Recall (useful):      {macro['mean_recall_useful']:.3f}",
        f"  Recall (overall):     {macro['mean_recall_overall']:.3f}",
        f"  Precision (known):    {macro['mean_precision_known']:.3f}",
        f"  Precision (strict):   {macro['mean_precision_strict']:.3f}",
        f"  F1 score:             {macro['mean_f1_score']:.3f}",
        f"  F1 (median / p10 / p90): "
        f"{macro['median_f1']:.3f} / {macro['p10_f1']:.3f} / {macro['p90_f1']:.3f}",
        f"  Mean output size:     {macro['mean_predicted_size']:.1f}",
        "",
        "--- MICRO METRICS (pooled across all samples) ---",
        f"  Recall (required):    {micro['recall_required']:.3f}  "
        f"({micro['total_tp_required']}/{micro['total_required']})",
        f"  Recall (useful):      {micro['recall_useful']:.3f}  "
        f"({micro['total_tp_useful']}/{micro['total_useful']})",
        f"  Recall (overall):     {micro['recall_overall']:.3f}",
        f"  Precision (known):    {micro['precision_known']:.3f}  "
        f"(FP={micro['total_fp_irrelevant']}, pred={micro['total_predicted']})",
        f"  F1 score:             {micro['f1_score']:.3f}",
        "",
        "--- WORST 5 SAMPLES (by F1) ---",
    ]
    for w in s["worst_samples"]:
        lines.append(
            f"  [F1={w['f1']:.2f}] rec_req={w['recall_required']:.2f} "
            f"prec={w['precision_known']:.2f}  {w['sample_id']}"
        )
        lines.append(f"    query: {w['query']}")
    lines.append("")
    lines.append("--- BEST 3 SAMPLES (by F1) ---")
    for b in s["best_samples"]:
        lines.append(
            f"  [F1={b['f1']:.2f}] rec_req={b['recall_required']:.2f}  {b['sample_id']}"
        )
        lines.append(f"    query: {b['query']}")
    lines.append("=" * 70)
    return "\n".join(lines)
