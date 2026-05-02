#!/usr/bin/env python3
"""CLI для подхода ``anchor_neighbors``.

Дизайн:

* На вход — slim-CSV `(repo, query)` (по умолчанию `data/dataset_queries.csv`).
  Pipeline видит ТОЛЬКО эти два поля.
* Эталон (required/useful) грузится отдельно (по умолчанию `data/dataset.csv`)
  и используется ИСКЛЮЧИТЕЛЬНО внутри `metrics.py` после выполнения этапа.
  Это структурно исключает утечку ground-truth в pipeline.
* Глубина прогона задаётся `--until {rag,anchor,neighbors,prune}`. Метрики
  считаются только для тех этапов, которые реально выполнились.
* Все параметры модели/ретривера/LLM — из `configs/config.yaml`. В этом
  скрипте никаких дефолтов нет.

Пример:

    # Только RAG, чтобы понять качество top-K кандидатов на всём датасете
    python src/approaches/anchor_neighbors/run.py --until rag

    # Полный прогон с ограничением на 5 запросов и без матчинга по эталону
    python src/approaches/anchor_neighbors/run.py --until prune --limit 5 --no-eval

    # Один конкретный (repo, query) для дебага одной строки
    python src/approaches/anchor_neighbors/run.py --until prune \\
        --repo apache/hadoop --query "..." 

Выходы (в `outputs_dir` из конфига, по умолчанию
`data/results/anchor_neighbors/`):
    report.jsonl              — поэтапные метрики и payload каждого этапа.
    aggregate.json            — усреднённые метрики по всем сэмплам.
    samples/<idx>.json        — финальный подграф для каждого сэмпла.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Сделать корень проекта импортируемым, когда этот скрипт запускают как файл.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.approaches.anchor_neighbors.debug_report import (  # noqa: E402
    JsonlReportWriter,
    aggregate,
)
from src.approaches.anchor_neighbors.ground_truth import (  # noqa: E402
    GoldTruthIndex,
)
from src.approaches.anchor_neighbors.metrics import (  # noqa: E402
    SampleReport,
    build_sample_report,
)
from src.approaches.anchor_neighbors.pipeline import (  # noqa: E402
    AnchorNeighborsPipeline,
    PipelineOutcome,
)
from src.approaches.anchor_neighbors.settings import (  # noqa: E402
    build_runner,
    load_settings,
)
from src.approaches.anchor_neighbors.stage_outputs import StageName  # noqa: E402
from src.core.config import load_config  # noqa: E402
from src.core.io import load_diagram, save_json  # noqa: E402
from src.core.types import ApproachInputs  # noqa: E402
from src.eval.annotations import diagram_filename_for_repo  # noqa: E402


# ---- модель сэмпла на входе ----------------------------------------------

@dataclass(frozen=True)
class SlimSample:
    """Только то, что pipeline разрешено видеть."""
    repo: str
    query: str


def load_slim_csv(path: Path) -> list[SlimSample]:
    """Прочитать CSV из `scripts/anonymize_dataset.py`. Колонки: repo,query."""
    if not path.exists():
        raise FileNotFoundError(
            f"slim-датасет не найден: {path}. "
            f"Сгенерируй его: `python scripts/anonymize_dataset.py`"
        )
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or set(reader.fieldnames) < {"repo", "query"}:
            raise ValueError(
                f"{path}: ожидаются колонки 'repo' и 'query', "
                f"нашёл: {reader.fieldnames}"
            )
        return [SlimSample(repo=r["repo"], query=r["query"]) for r in reader]


# ---- разбор флагов -------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Запуск подхода 'anchor_neighbors' по slim-датасету.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--until",
        choices=[s.value for s in StageName],
        default=StageName.PRUNE.value,
        help="До какого этапа прогонять: rag/anchor/neighbors/prune.",
    )
    p.add_argument(
        "--queries",
        type=Path,
        default=Path("data/dataset_queries.csv"),
        help="CSV (repo,query) — ровно то, что подаётся в pipeline.",
    )
    p.add_argument(
        "--gold",
        type=Path,
        default=Path("data/dataset.csv"),
        help="Полный датасет с аннотациями — нужен ТОЛЬКО для метрик.",
    )
    p.add_argument(
        "--diagrams-dir",
        type=Path,
        default=Path("data/diagrams_normalized"),
        help="Папка с нормализованными диаграммами.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="YAML-конфиг проекта.",
    )
    p.add_argument(
        "--repo",
        default="",
        help="Оставить только сэмплы этого repo (например, apache/hadoop).",
    )
    p.add_argument(
        "--query",
        default="",
        help="Оставить только сэмпл с точно таким текстом запроса.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Сколько сэмплов обработать (0 = все).",
    )
    p.add_argument(
        "--no-eval",
        action="store_true",
        help="Не считать метрики (нужно только для прогона без эталона).",
    )
    return p.parse_args()


# ---- основная логика -----------------------------------------------------

async def _process_one(
    pipeline: AnchorNeighborsPipeline,
    sample: SlimSample,
    diagrams_dir: Path,
    diagram_cache: dict[str, dict],
    until: StageName,
    sample_idx: int,
) -> PipelineOutcome:
    """Прогнать pipeline по одному сэмплу."""
    fname = diagram_filename_for_repo(sample.repo)
    if fname not in diagram_cache:
        diagram_cache[fname] = load_diagram(diagrams_dir / fname)
    diagram = diagram_cache[fname]

    inputs = ApproachInputs(
        query=sample.query,
        diagram=diagram,
        # sample_id используется ТОЛЬКО для имён файлов, pipeline на нём
        # не строит решений — это закреплено в комментариях ApproachInputs.
        sample_id=f"sample_{sample_idx:04d}",
        repo=sample.repo,
    )
    return await pipeline.run_with_stages(inputs, until=until)


def _filter_samples(
    samples: list[SlimSample], repo: str, query: str, limit: int
) -> list[SlimSample]:
    """Применить фильтры командной строки к slim-датасету."""
    out = samples
    if repo:
        out = [s for s in out if s.repo == repo]
    if query:
        out = [s for s in out if s.query == query]
    if limit:
        out = out[:limit]
    return out


async def _run_async(args: argparse.Namespace) -> int:
    cfg = load_config(str(args.config))
    settings = load_settings(cfg)
    pipeline = build_runner(cfg)

    # 1. Что подаём в пайплайн.
    samples = _filter_samples(
        load_slim_csv(args.queries),
        repo=args.repo,
        query=args.query,
        limit=args.limit,
    )
    if not samples:
        print("[error] под фильтры не попало ни одного сэмпла.", file=sys.stderr)
        return 2

    # 2. Эталон загружается отдельно. Pipeline его не видит.
    gold_index = None
    missing_gold: list[tuple[str, str]] = []
    if not args.no_eval:
        gold_index = GoldTruthIndex.from_csv(args.gold)
        if gold_index.duplicates:
            print(
                f"[warn] эталон содержит {len(gold_index.duplicates)} "
                f"дублирующих ключей (repo, query); берём первый.",
                file=sys.stderr,
            )

    # 3. Подготовка путей вывода.
    out_dir: Path = settings.pipeline.outputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    report_path = out_dir / "report.jsonl"
    aggregate_path = out_dir / "aggregate.json"

    until = StageName(args.until)
    diagram_cache: dict[str, dict] = {}
    sample_reports: list[SampleReport] = []
    runtime_errors: list[dict] = []

    started = time.time()
    with JsonlReportWriter(report_path) as report:
        for idx, sample in enumerate(samples):
            try:
                outcome = await _process_one(
                    pipeline, sample, args.diagrams_dir,
                    diagram_cache, until, idx,
                )
            except Exception as e:  # noqa: BLE001 — единая граница CLI
                runtime_errors.append({
                    "index": idx, "repo": sample.repo, "query": sample.query,
                    "error": repr(e),
                })
                continue

            # Сохраним финальный подграф этого сэмпла на диск.
            save_json(
                outcome.result.to_diagram(),
                samples_dir / f"sample_{idx:04d}.json",
            )

            # Метрики: только если у нас есть эталон по этому ключу.
            if gold_index is not None:
                gold = gold_index.lookup(sample.repo, sample.query)
                if gold is None:
                    missing_gold.append((sample.repo, sample.query))
                    continue
                rep = build_sample_report(
                    sample_id=gold.sample_id or f"sample_{idx:04d}",
                    repo=sample.repo,
                    query=sample.query,
                    stages=outcome.stages,
                    gold=gold,
                )
                sample_reports.append(rep)
                report.write(rep)

    elapsed = time.time() - started

    # 4. Свести метрики и записать сводку.
    if sample_reports:
        agg = aggregate(sample_reports)
        agg["until"] = until.value
        agg["elapsed_s"] = round(elapsed, 2)
        save_json(agg, aggregate_path)
        print(json.dumps(agg, ensure_ascii=False, indent=2))

    print(
        f"\nready: samples={len(samples)} processed={len(sample_reports)} "
        f"errors={len(runtime_errors)} missing_gold={len(missing_gold)} "
        f"elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )
    if runtime_errors:
        save_json(runtime_errors, out_dir / "errors.json")
    if missing_gold:
        save_json(missing_gold, out_dir / "missing_gold.json")
    return 0


def main() -> int:
    return asyncio.run(_run_async(parse_args()))


if __name__ == "__main__":
    sys.exit(main())
