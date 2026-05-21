#!/usr/bin/env python3
"""Оценка результатов human_like_agent подхода.

Использование:
    python evaluate.py
    python evaluate.py --results-dir custom/path
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Определить пути
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "human_like_agent_1"
DATASET_PATH = PROJECT_ROOT / "data" / "dataset_1_iter_subtract.csv"


def load_dataset() -> dict[str, dict]:
    """Загрузить ground truth из dataset.csv.
    
    Returns:
        Dict: {sample_id: {repo, query, central_node, entity_annotations}}
    """
    dataset = {}
    
    with DATASET_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset[row["sample_id"]] = row
    
    return dataset


def parse_entity_annotations(annotations_str: str) -> dict[str, str]:
    """Парсить entity_annotations из строки JSON.
    
    Args:
        annotations_str: JSON строка вида '{"Class1": "required", "Class2": "useful"}'
        
    Returns:
        Dict: {node_id: label}
    """
    try:
        return json.loads(annotations_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def calculate_metrics(predicted: dict, gold: dict[str, str]) -> dict:
    """Вычислить метрики.
    
    Args:
        predicted: {"required": [...], "useful": [...]}
        gold: {node_id: "required" | "useful"}
        
    Returns:
        Dict с метриками: precision, recall_required, recall_useful, recall_overall, f1_score
    """
    pred_required = set(predicted.get("required", []))
    pred_useful = set(predicted.get("useful", []))
    pred_all = pred_required | pred_useful
    
    gold_required = {k for k, v in gold.items() if v == "required"}
    gold_useful = {k for k, v in gold.items() if v == "useful"}
    gold_all = set(gold.keys())
    
    # Precision
    if len(pred_all) > 0:
        precision = len(pred_all & gold_all) / len(pred_all)
    else:
        precision = 0.0
    
    # Recall (required)
    if len(gold_required) > 0:
        recall_required = len(pred_required & gold_required) / len(gold_required)
    else:
        recall_required = 0.0
    
    # Recall (useful)
    if len(gold_useful) > 0:
        recall_useful = len(pred_useful & gold_useful) / len(gold_useful)
    else:
        recall_useful = 0.0
    
    # Recall (overall)
    if len(gold_all) > 0:
        recall_overall = len(pred_all & gold_all) / len(gold_all)
    else:
        recall_overall = 0.0
    
    # F1
    if precision + recall_overall > 0:
        f1_score = 2 * (precision * recall_overall) / (precision + recall_overall)
    else:
        f1_score = 0.0
    
    return {
        "precision": precision,
        "recall_required": recall_required,
        "recall_useful": recall_useful,
        "recall_overall": recall_overall,
        "f1_score": f1_score,
        "total_predicted": len(pred_all),
        "total_gold": len(gold_all),
    }


def evaluate_results(results_dir: Path, dataset: dict[str, dict]) -> dict:
    """Оценить все результаты.
    
    Args:
        results_dir: Путь к директории с результатами
        dataset: Ground truth из dataset.csv
        
    Returns:
        Dict с агрегированными метриками и per-sample результатами
    """
    per_sample = []
    
    for result_file in sorted(results_dir.glob("*.json")):
        # Извлечь sample_id из имени файла
        # Формат: {repo}__{sample_id}.json
        filename = result_file.stem
        parts = filename.split("__")
        print(parts)
        if len(parts) != 2:
            print(f"Warning: Skipping {result_file.name} (unexpected format)")
            continue
        
        sample_id = parts[1]
        
        if sample_id not in dataset:
            print(f"Warning: Sample {sample_id} not found in dataset")
            continue
        
        # Загрузить результат
        with result_file.open() as f:
            predicted = json.load(f)
        
        # Загрузить ground truth
        gold_annotations = parse_entity_annotations(
            dataset[sample_id]["entity_annotations"]
        )
        
        # Вычислить метрики
        metrics = calculate_metrics(predicted, gold_annotations)
        
        per_sample.append({
            "sample_id": sample_id,
            "repo": dataset[sample_id]["repo"],
            **metrics
        })
    
    # Агрегированные метрики
    if per_sample:
        summary = {
            "precision": sum(s["precision"] for s in per_sample) / len(per_sample),
            "recall_required": sum(s["recall_required"] for s in per_sample) / len(per_sample),
            "recall_useful": sum(s["recall_useful"] for s in per_sample) / len(per_sample),
            "recall_overall": sum(s["recall_overall"] for s in per_sample) / len(per_sample),
            "f1_score": sum(s["f1_score"] for s in per_sample) / len(per_sample),
            "total_samples": len(per_sample),
        }
    else:
        summary = {}
    
    return {
        "summary": summary,
        "per_sample": per_sample,
    }


def parse_args() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    p = argparse.ArgumentParser(
        description="Evaluate human_like_agent results against ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Path to results directory",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to dataset.csv with ground truth",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="Path to save evaluation report JSON (optional)",
    )
    return p.parse_args()


def main():
    """Основная логика."""
    args = parse_args()
    
    print("="*60)
    print("Human-Like Agent: Evaluation")
    print("="*60)
    
    # Загрузить dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} samples")
    
    # Оценить результаты
    print(f"Evaluating results in: {args.results_dir}")
    evaluation = evaluate_results(args.results_dir, dataset)
    
    if not evaluation["per_sample"]:
        print("ERROR: No results found to evaluate!")
        return
    
    # Вывести сводку
    summary = evaluation["summary"]
    print("\n" + "="*60)
    print("SUMMARY METRICS:")
    print("="*60)
    print(f"Total samples:     {summary['total_samples']}")
    print(f"Precision:         {summary['precision']:.3f}")
    print(f"Recall Required:   {summary['recall_required']:.3f}")
    print(f"Recall Useful:     {summary['recall_useful']:.3f}")
    print(f"Recall Overall:    {summary['recall_overall']:.3f}")
    print(f"F1 Score:          {summary['f1_score']:.3f}")
    
    # Per-sample детали
    print("\n" + "="*60)
    print("PER-SAMPLE RESULTS:")
    print("="*60)
    print(f"{'Sample ID':<20} {'Repo':<20} {'Prec':<6} {'Rec':<6} {'F1':<6}")
    print("-"*60)
    
    for s in evaluation["per_sample"]:
        print(f"{s['sample_id']:<20} {s['repo']:<20} "
              f"{s['precision']:.3f}  {s['recall_overall']:.3f}  {s['f1_score']:.3f}")
    
    # Сохранить отчёт
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(evaluation, f, indent=2)
        print(f"\nEvaluation report saved to: {args.output}")
    else:
        # Сохранить рядом с результатами
        report_path = args.results_dir / "evaluation_report.json"
        with report_path.open("w") as f:
            json.dump(evaluation, f, indent=2)
        print(f"\nEvaluation report saved to: {report_path}")
    
    print("="*60)


if __name__ == "__main__":
    main()
