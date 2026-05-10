#!/usr/bin/env python3
"""CLI для подхода human_like_agent.

Запуск:
    # Один сэмпл
    python src/approaches/human_like_agent/run.py --limit 1

    # Конкретный репозиторий
    python src/approaches/human_like_agent/run.py --repo apache/hadoop

    # С оценкой качества
    python src/approaches/human_like_agent/run.py --repo apache/flink --eval

    # Конкретный query (для дебага)
    python src/approaches/human_like_agent/run.py --repo apache/hadoop --query "Show file system classes"
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Сделать корень проекта импортируемым
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.approaches.human_like_agent.runner import HumanLikeAgentRunner  # noqa: E402
from src.approaches.human_like_agent.settings import load_settings, make_llm_client  # noqa: E402
from src.core.config import load_config  # noqa: E402
from src.core.logger import get_logger  # noqa: E402
from src.eval.evaluator import evaluate_sample  # noqa: E402

logger = get_logger(__name__)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Загрузить dataset.csv с полями: sample_id, repo, query, central_node, entity_annotations."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset не найден: {path}. Создай его: python scripts/build_dataset.py"
        )
    
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Запуск human_like_agent подхода",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/dataset.csv"),
        help="CSV с датасетом (sample_id, repo, query, annotations)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="YAML конфигурация",
    )
    p.add_argument(
        "--repo",
        default="",
        help="Фильтр по репозиторию (например, apache/hadoop)",
    )
    p.add_argument(
        "--query",
        default="",
        help="Фильтр по точному тексту запроса",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Ограничить количество сэмплов (0 = все)",
    )
    p.add_argument(
        "--eval",
        action="store_true",
        help="Вычислить метрики (Precision, Recall, F1)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Подробный вывод (debug уровень логов)",
    )
    return p.parse_args()


async def run_sample(
    runner: HumanLikeAgentRunner,
    sample: dict[str, Any],
    output_dir: Path,
    eval_mode: bool = False,
) -> dict[str, Any]:
    """Запустить подход на одном сэмпле.
    
    Args:
        runner: HumanLikeAgentRunner instance
        sample: Строка из dataset.csv
        output_dir: Куда сохранить результат
        eval_mode: Вычислить метрики
        
    Returns:
        Dict с результатом и метриками (если eval_mode=True)
    """
    sample_id = sample["sample_id"]
    repo = sample["repo"]
    
    try:
        # Запустить агента
        result = await runner.run_async(sample)
        
        # Сохранить результат
        output_file = output_dir / f"{repo.replace('/', '__')}__{sample_id}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "sample_id": sample_id,
            "repo": repo,
            "query": sample["query"],
            "required": result.get("required", []),
            "useful": result.get("useful", []),
        }
        
        with output_file.open("w") as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"[{sample_id}] Saved to {output_file}")
        
        # Вычислить метрики (если eval_mode)
        metrics = {}
        if eval_mode:
            try:
                metrics = evaluate_sample(
                    result_path=output_file,
                    gold_required=sample.get("entity_annotations", "{}"),
                    central_node=sample.get("central_node", ""),
                )
                logger.info(
                    f"[{sample_id}] Metrics: "
                    f"P={metrics.get('precision', 0):.3f} "
                    f"R={metrics.get('recall_overall', 0):.3f} "
                    f"F1={metrics.get('f1_score', 0):.3f}"
                )
            except Exception as e:
                logger.warning(f"[{sample_id}] Metrics failed: {e}")
        
        return {
            "sample_id": sample_id,
            "repo": repo,
            "success": True,
            "result": output_data,
            "metrics": metrics,
        }
        
    except Exception as e:
        logger.error(f"[{sample_id}] Failed: {e}", exc_info=True)
        return {
            "sample_id": sample_id,
            "repo": repo,
            "success": False,
            "error": str(e),
        }


async def main():
    args = parse_args()
    
    # Настроить логирование
    if args.verbose:
        import logging
        logging.getLogger("src").setLevel(logging.DEBUG)
    
    # Загрузить конфигурацию
    cfg = load_config(args.config)
    settings = load_settings(cfg)
    llm = make_llm_client(settings.llm)
    
    # Создать runner
    runner = HumanLikeAgentRunner(settings, llm)
    
    # Загрузить датасет
    dataset = load_dataset(args.dataset)
    logger.info(f"Loaded {len(dataset)} samples from {args.dataset}")
    
    # Фильтры
    if args.repo:
        dataset = [s for s in dataset if s["repo"] == args.repo]
        logger.info(f"Filtered to repo={args.repo}: {len(dataset)} samples")
    
    if args.query:
        dataset = [s for s in dataset if s["query"] == args.query]
        logger.info(f"Filtered to query match: {len(dataset)} samples")
    
    if args.limit > 0:
        dataset = dataset[:args.limit]
        logger.info(f"Limited to {args.limit} samples")
    
    if not dataset:
        logger.error("No samples to process after filtering!")
        return
    
    # Выходная директория
    output_dir = settings.outputs_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Обработать все сэмплы
    results = []
    for sample in tqdm(dataset, desc="Processing samples"):
        result = await run_sample(runner, sample, output_dir, eval_mode=args.eval)
        results.append(result)
    
    # Сводная статистика
    success_count = sum(1 for r in results if r["success"])
    logger.info(f"\n{'='*60}")
    logger.info(f"Processed: {len(results)} samples")
    logger.info(f"Success: {success_count}/{len(results)}")
    logger.info(f"Failed: {len(results) - success_count}")
    
    # Агрегированные метрики (если eval_mode)
    if args.eval:
        metrics_list = [r["metrics"] for r in results if r["success"] and r.get("metrics")]
        if metrics_list:
            avg_metrics = {
                "precision": sum(m.get("precision", 0) for m in metrics_list) / len(metrics_list),
                "recall_overall": sum(m.get("recall_overall", 0) for m in metrics_list) / len(metrics_list),
                "f1_score": sum(m.get("f1_score", 0) for m in metrics_list) / len(metrics_list),
            }
            logger.info(f"\n{'='*60}")
            logger.info("AGGREGATE METRICS:")
            logger.info(f"  Precision:      {avg_metrics['precision']:.3f}")
            logger.info(f"  Recall Overall: {avg_metrics['recall_overall']:.3f}")
            logger.info(f"  F1 Score:       {avg_metrics['f1_score']:.3f}")
            
            # Сохранить агрегированный отчёт
            report_file = output_dir / "evaluation_report.json"
            with report_file.open("w") as f:
                json.dump({
                    "summary": avg_metrics,
                    "total_samples": len(results),
                    "success_count": success_count,
                    "per_sample": results,
                }, f, indent=2)
            logger.info(f"\nEvaluation report saved to: {report_file}")
    
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
