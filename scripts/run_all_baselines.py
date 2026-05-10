#!/usr/bin/env python3
"""Запустить все бейзлайны на датасете и сгенерировать отчеты.

Этот скрипт автоматизирует запуск всех query-agnostic и lexical бейзлайнов,
а также oracle бейзлайнов. Результаты сохраняются в data/results/ и можно
легко сравнить с основными подходами.

Запускаемые бейзлайны:
    Query-agnostic:
        - empty          : пустой граф (recall=0)
        - full_diagram   : весь граф (recall=1.0, низкая precision)
        - random_subset  : случайный набор узлов
        - top_degree     : узлы с наибольшей степенью
    
    Lexical:
        - bm25           : BM25 ранжирование по запросу
    
    Oracle (видят ground truth):
        - central_plus_neighbors : центральный узел + соседи 1-го уровня
        - gold_only              : точный золотой ответ (F1=1.0)

Использование:
    # Запустить все бейзлайны
    python scripts/run_all_baselines.py
    
    # Только query-agnostic бейзлайны
    python scripts/run_all_baselines.py --skip-bm25 --skip-oracle
    
    # Ограничить количество сэмплов
    python scripts/run_all_baselines.py --limit 10
    
    # Конкретный репозиторий
    python scripts/run_all_baselines.py --repo apache/hadoop

Результаты:
    - data/results/<baseline_name>/<sample_id>.json
    - data/results/<baseline_name>/evaluation_report.json
    - data/results/baselines_summary.json (сводка всех бейзлайнов)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.io import save_json  # noqa: E402


# Список всех query-agnostic и lexical бейзлайнов
STANDARD_BASELINES = [
    "empty",
    "full_diagram", 
    "random_subset",
    "top_degree",
    "bm25",
]

# Oracle бейзлайны (видят ground truth)
ORACLE_BASELINES = [
    "central_plus_neighbors",
    "gold_only",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Запустить все бейзлайны и создать сводный отчет.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset",
        default="data/dataset.csv",
        help="Путь к датасету (default: data/dataset.csv)",
    )
    p.add_argument(
        "--diagrams-dir",
        default="data/diagrams_normalized",
        help="Директория с нормализованными диаграммами (default: data/diagrams_normalized)",
    )
    p.add_argument(
        "--output-root",
        default="data/results",
        help="Корневая директория для результатов (default: data/results)",
    )
    p.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Конфигурационный файл (default: configs/config.yaml)",
    )
    p.add_argument(
        "--repo",
        default="",
        help="Запустить только для этого репозитория (например, apache/hadoop)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Максимальное количество сэмплов (0 = все)",
    )
    p.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Пропустить BM25 бейзлайн (если rank_bm25 не установлен)",
    )
    p.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Пропустить oracle бейзлайны",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Перезаписать существующие результаты",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Подробный вывод",
    )
    return p.parse_args()


def run_baseline(
    baseline: str,
    dataset: str,
    diagrams_dir: str,
    output_root: str,
    config: str,
    repo: str = "",
    limit: int = 0,
    overwrite: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Запустить один стандартный бейзлайн через scripts/run.py"""
    cmd = [
        sys.executable,
        "scripts/run.py",
        "--approach", baseline,
        "--dataset", dataset,
        "--diagrams-dir", diagrams_dir,
        "--output-dir", f"{output_root}/{baseline}",
        "--config", config,
    ]
    
    if repo:
        cmd.extend(["--repo", repo])
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    if overwrite:
        cmd.append("--overwrite")
    if verbose:
        cmd.append("--verbose")
    
    print(f"\n{'='*60}")
    print(f"Running baseline: {baseline}")
    print(f"{'='*60}")
    
    start = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=not verbose)
    elapsed = time.time() - start
    
    # Загружаем отчет об оценке
    report_path = Path(output_root) / baseline / "evaluation_report.json"
    report_data = {}
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as f:
            report_data = json.load(f)
    
    return {
        "baseline": baseline,
        "success": result.returncode == 0,
        "elapsed_s": round(elapsed, 2),
        "evaluation": report_data.get("summary", {}),
        "command": " ".join(cmd),
    }


def run_oracle_baselines(
    dataset: str,
    diagrams_dir: str,
    output_root: str,
    repo: str = "",
    limit: int = 0,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Запустить все oracle бейзлайны через scripts/run_oracle_baselines.py"""
    cmd = [
        sys.executable,
        "scripts/run_oracle_baselines.py",
        "--baselines", *ORACLE_BASELINES,
        "--dataset", dataset,
        "--diagrams-dir", diagrams_dir,
        "--output-root", output_root,
    ]
    
    if repo:
        cmd.extend(["--repo", repo])
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    
    print(f"\n{'='*60}")
    print("Running oracle baselines")
    print(f"{'='*60}")
    
    start = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=not verbose)
    elapsed = time.time() - start
    
    # Загружаем отчеты для каждого oracle бейзлайна
    results = []
    for baseline in ORACLE_BASELINES:
        report_path = Path(output_root) / f"oracle_{baseline}" / "evaluation_report.json"
        report_data = {}
        if report_path.exists():
            with report_path.open("r", encoding="utf-8") as f:
                report_data = json.load(f)
        
        results.append({
            "baseline": f"oracle_{baseline}",
            "success": result.returncode == 0,
            "elapsed_s": round(elapsed / len(ORACLE_BASELINES), 2),
            "evaluation": report_data.get("summary", {}),
        })
    
    return results


async def main_async(args: argparse.Namespace) -> None:
    """Главная функция запуска всех бейзлайнов"""
    
    # Проверяем наличие необходимых файлов
    dataset_path = Path(args.dataset)
    diagrams_path = Path(args.diagrams_dir)
    
    if not dataset_path.exists():
        print(f"[error] Датасет не найден: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    
    if not diagrams_path.exists():
        print(f"[error] Директория диаграмм не найдена: {diagrams_path}", file=sys.stderr)
        sys.exit(1)
    
    # Подготовка списка бейзлайнов
    baselines_to_run = [b for b in STANDARD_BASELINES]
    if args.skip_bm25:
        baselines_to_run = [b for b in baselines_to_run if b != "bm25"]
    
    # Запуск стандартных бейзлайнов
    all_results = []
    
    print("\n" + "="*60)
    print("ЗАПУСК СТАНДАРТНЫХ БЕЙЗЛАЙНОВ")
    print("="*60)
    
    for baseline in baselines_to_run:
        result = run_baseline(
            baseline=baseline,
            dataset=args.dataset,
            diagrams_dir=args.diagrams_dir,
            output_root=args.output_root,
            config=args.config,
            repo=args.repo,
            limit=args.limit,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        all_results.append(result)
    
    # Запуск oracle бейзлайнов
    if not args.skip_oracle:
        print("\n" + "="*60)
        print("ЗАПУСК ORACLE БЕЙЗЛАЙНОВ")
        print("="*60)
        
        oracle_results = run_oracle_baselines(
            dataset=args.dataset,
            diagrams_dir=args.diagrams_dir,
            output_root=args.output_root,
            repo=args.repo,
            limit=args.limit,
            verbose=args.verbose,
        )
        all_results.extend(oracle_results)
    
    # Создание сводного отчета
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "diagrams_dir": args.diagrams_dir,
        "filters": {
            "repo": args.repo or "all",
            "limit": args.limit or "all",
        },
        "baselines": all_results,
    }
    
    summary_path = Path(args.output_root) / "baselines_summary.json"
    save_json(summary, summary_path)
    
    # Вывод сводной таблицы
    print("\n" + "="*60)
    print("СВОДНЫЕ РЕЗУЛЬТАТЫ")
    print("="*60)
    print(f"\n{'Baseline':<30} {'Success':<10} {'F1':<10} {'Prec':<10} {'Rec':<10} {'Time(s)':<10}")
    print("-" * 80)
    
    for r in all_results:
        name = r["baseline"]
        success = "✓" if r["success"] else "✗"
        eval_data = r.get("evaluation", {})
        f1 = f"{eval_data.get('f1_score', 0):.3f}" if eval_data else "N/A"
        prec = f"{eval_data.get('precision', 0):.3f}" if eval_data else "N/A"
        rec = f"{eval_data.get('recall_overall', 0):.3f}" if eval_data else "N/A"
        elapsed = f"{r['elapsed_s']:.1f}"
        
        print(f"{name:<30} {success:<10} {f1:<10} {prec:<10} {rec:<10} {elapsed:<10}")
    
    print("\n" + "="*60)
    print(f"Сводный отчет сохранен: {summary_path}")
    print("="*60 + "\n")


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
