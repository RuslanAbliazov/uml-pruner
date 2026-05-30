#!/usr/bin/env python3
"""CLI для подхода ``anchor_neighbors``.

Два режима:
1. Полный прогон (этапы 1–4):
       python src/approaches/anchor_neighbors/run.py [--limit N] [--repo R] [--until STAGE]
   При `--until neighbors` подграфы после этапа 3 сохраняются в
   `<outputs_dir>/stage3/<sample_id>.json` — для последующего запуска этапа 4.

2. Только LLM-прунинг (этап 4) по готовым данным:
       python src/approaches/anchor_neighbors/run.py --from-stage3 <dir>

Результаты пишутся в `<outputs_dir>/` (из конфига, по умолчанию
`data/results/anchor_neighbors/<selector>/`) в формате:
    {"sample_id": ..., "repo": ..., "query": ..., "required": [...], "useful": [...]}
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
import traceback as tb
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.approaches.anchor_neighbors.debug_report import JsonlReportWriter, aggregate  # noqa: E402
from src.approaches.anchor_neighbors.ground_truth import GoldTruthIndex  # noqa: E402
from src.approaches.anchor_neighbors.metrics import SampleReport, build_sample_report  # noqa: E402
from src.approaches.anchor_neighbors.pipeline import AnchorNeighborsPipeline, PipelineOutcome  # noqa: E402
from src.approaches.anchor_neighbors.settings import build_runner, load_settings  # noqa: E402
from src.approaches.anchor_neighbors.stage_outputs import StageName  # noqa: E402
from src.core.config import load_config  # noqa: E402
from src.core.io import load_diagram, save_json  # noqa: E402
from src.core.types import ApproachInputs  # noqa: E402
from src.eval.annotations import diagram_filename_for_repo  # noqa: E402


@dataclass(frozen=True)
class SlimSample:
    """Только то, что pipeline разрешено видеть."""
    repo: str
    query: str


def load_slim_csv(path: Path) -> list[SlimSample]:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Запуск подхода 'anchor_neighbors'.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--until",
        choices=[s.value for s in StageName],
        default=StageName.PRUNE.value,
        help="До какого этапа прогонять (rag/anchor/neighbors/prune).",
    )
    p.add_argument(
        "--queries",
        type=Path,
        default=Path("data/dataset_queries.csv"),
        help="CSV (repo, query) — вход для pipeline.",
    )
    p.add_argument(
        "--gold",
        type=Path,
        default=Path("data/dataset.csv"),
        help="Полный датасет с аннотациями — только для метрик.",
    )
    p.add_argument(
        "--diagrams-dir",
        type=Path,
        default=Path("data/diagrams_normalized"),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
    )
    p.add_argument("--repo", default="", help="Фильтр по репозиторию.")
    p.add_argument("--query", default="", help="Фильтр по точному тексту запроса.")
    p.add_argument("--limit", type=int, default=0, help="Максимум сэмплов (0 = все).")
    p.add_argument("--no-eval", action="store_true", help="Не считать метрики.")
    p.add_argument(
        "--from-stage3",
        type=Path,
        default=None,
        help="Директория с данными этапа 3 ({sample_id}.json). "
             "Пропускает этапы 1–3 и запускает только LLM-прунинг.",
    )
    return p.parse_args()


def _flat_result(sample_id: str, repo: str, query: str, outcome: PipelineOutcome) -> dict:
    return {
        "sample_id": sample_id,
        "repo": repo,
        "query": query,
        "required": sorted(outcome.result.required_node_ids),
        "useful": sorted(outcome.result.useful_node_ids),
    }


def _filter_samples(samples: list[SlimSample], repo: str, query: str, limit: int) -> list[SlimSample]:
    if repo:
        samples = [s for s in samples if s.repo == repo]
    if query:
        samples = [s for s in samples if s.query == query]
    if limit:
        samples = samples[:limit]
    return samples


async def _run_from_stage3(
    args: argparse.Namespace,
    pipeline: AnchorNeighborsPipeline,
    settings,
) -> int:
    """Этап 4 (LLM-прунинг) по готовым данным из директории stage3."""
    from src.approaches.anchor_neighbors import stage4_prune

    stage3_dir = args.from_stage3
    if not stage3_dir.exists():
        print(f"[error] {stage3_dir} не существует.", file=sys.stderr)
        return 2

    files = sorted(stage3_dir.glob("*.json"))
    if not files:
        print(f"[error] В {stage3_dir} нет JSON файлов.", file=sys.stderr)
        return 2
    if args.limit:
        files = files[:args.limit]

    out_dir: Path = settings.pipeline.outputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_index = GoldTruthIndex.from_csv(args.gold) if not args.no_eval else None
    sample_reports: list[SampleReport] = []
    missing_gold: list[tuple[str, str]] = []
    runtime_errors: list[dict] = []

    started = time.time()
    with JsonlReportWriter(out_dir / "report.jsonl") as report:
        for idx, path in tqdm(enumerate(files), total=len(files),
                               desc="from-stage3", unit="sample"):
            try:
                with path.open("r", encoding="utf-8") as f:
                    d = json.load(f)
                sample_id = d.get("sample_id", f"sample_{idx:04d}")
                repo = d.get("repo", "")
                query = d["query"]
                s4 = await stage4_prune.prune_subgraph(
                    query=query,
                    sub_nodes=d["sub_nodes"],
                    sub_edges=d["sub_edges"],
                    llm=pipeline._llm,
                    tracer=pipeline._tracer,
                    sample_id=sample_id,
                )
            except Exception as e:
                runtime_errors.append({
                    "file": str(path), "error": str(e),
                    "traceback": "".join(tb.format_exception(type(e), e, e.__traceback__)),
                })
                print(f"[error] {path.name}: {e}", file=sys.stderr)
                continue

            required = sorted(s4.payload.get("required", [])) if s4.is_ok() else []
            useful = sorted(s4.payload.get("useful", [])) if s4.is_ok() else []
            save_json(
                {"sample_id": sample_id, "repo": repo, "query": query,
                 "required": required, "useful": useful},
                out_dir / f"{sample_id}.json",
            )

            if gold_index is not None:
                gold = gold_index.lookup(repo, query)
                if gold is None:
                    missing_gold.append((repo, query))
                    continue
                rep = build_sample_report(
                    sample_id=sample_id, repo=repo, query=query,
                    stages={StageName.PRUNE: s4}, gold=gold,
                )
                sample_reports.append(rep)
                report.write(rep)

    elapsed = time.time() - started
    if sample_reports:
        agg = {**aggregate(sample_reports), "from_stage3": True,
               "elapsed_s": round(elapsed, 2)}
        save_json(agg, out_dir / "aggregate.json")
        print(json.dumps(agg, ensure_ascii=False, indent=2))

    print(
        f"\nready: files={len(files)} ok={len(sample_reports)} "
        f"errors={len(runtime_errors)} missing_gold={len(missing_gold)} "
        f"elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )
    if runtime_errors:
        save_json(runtime_errors, out_dir / "errors.json")
    if missing_gold:
        save_json(missing_gold, out_dir / "missing_gold.json")
    return 0


async def _run_async(args: argparse.Namespace) -> int:
    cfg = load_config(str(args.config))
    settings = load_settings(cfg)
    pipeline = build_runner(cfg)

    if args.from_stage3:
        return await _run_from_stage3(args, pipeline, settings)

    samples = _filter_samples(
        load_slim_csv(args.queries), args.repo, args.query, args.limit
    )
    if not samples:
        print("[error] под фильтры не попало ни одного сэмпла.", file=sys.stderr)
        return 2

    gold_index = None
    missing_gold: list[tuple[str, str]] = []
    if not args.no_eval:
        gold_index = GoldTruthIndex.from_csv(args.gold)
        if gold_index.duplicates:
            print(
                f"[warn] {len(gold_index.duplicates)} дублирующих ключей в эталоне.",
                file=sys.stderr,
            )

    out_dir: Path = settings.pipeline.outputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    until = StageName(args.until)
    diagram_cache: dict[str, dict] = {}
    sample_reports: list[SampleReport] = []
    runtime_errors: list[dict] = []

    started = time.time()
    with JsonlReportWriter(out_dir / "report.jsonl") as report:
        for idx, sample in tqdm(enumerate(samples), total=len(samples),
                                  desc="Processing", unit="sample"):
            fname = diagram_filename_for_repo(sample.repo)
            if fname not in diagram_cache:
                diagram_cache[fname] = load_diagram(args.diagrams_dir / fname)

            inputs = ApproachInputs(
                query=sample.query,
                diagram=diagram_cache[fname],
                sample_id=f"sample_{idx:04d}",
                repo=sample.repo,
            )

            try:
                outcome = await pipeline.run_with_stages(inputs, until=until)
            except Exception as e:  # noqa: BLE001
                runtime_errors.append({
                    "index": idx, "repo": sample.repo,
                    "query": sample.query, "error": repr(e),
                })
                continue

            # При --until neighbors финальных результатов нет — только stage3-файлы,
            # которые pipeline уже сохранил. Идём дальше.
            if until == StageName.NEIGHBORS:
                continue

            save_json(
                _flat_result(inputs.sample_id, sample.repo, sample.query, outcome),
                out_dir / f"{inputs.sample_id}.json",
            )

            if gold_index is not None:
                gold = gold_index.lookup(sample.repo, sample.query)
                if gold is None:
                    missing_gold.append((sample.repo, sample.query))
                    continue
                rep = build_sample_report(
                    sample_id=gold.sample_id or inputs.sample_id,
                    repo=sample.repo,
                    query=sample.query,
                    stages=outcome.stages,
                    gold=gold,
                )
                sample_reports.append(rep)
                report.write(rep)

    elapsed = time.time() - started
    if sample_reports:
        agg = {**aggregate(sample_reports), "until": until.value,
               "elapsed_s": round(elapsed, 2)}
        save_json(agg, out_dir / "aggregate.json")
        print(json.dumps(agg, ensure_ascii=False, indent=2))

    print(
        f"\nready: samples={len(samples)} ok={len(sample_reports)} "
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
