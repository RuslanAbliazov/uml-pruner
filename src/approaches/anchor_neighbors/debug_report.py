"""Запись поэтапного отчёта в JSONL.

Один файл на запуск. Одна строка JSON на сэмпл. Структура строки совпадает с
`SampleReport.to_dict()`:

    {
      "sample_id": "...",
      "repo": "...",
      "query": "...",
      "metrics": [ {stage, ok, aborted, size, ..., precision, recall, f1, ...},
                   ... ],   # по одной записи на каждый выполненный этап
      "stages":  { "rag": {...}, "anchor": {...}, ... }   # сырьё этапов
    }

Файл аппендится: можно добавлять записи по мере обработки сэмплов и
смотреть прогресс хвостом (`tail -f`). Заголовка нет — это JSONL.

Также есть `aggregate(reports)` — сводка по всем сэмплам с усреднёнными
метриками на каждом этапе. Полезна, когда хочется одной строкой увидеть,
«где именно падает recall»: на retrieve, anchor, neighbors или prune.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Iterable

from src.approaches.anchor_neighbors.metrics import SampleReport
from src.approaches.anchor_neighbors.stage_outputs import STAGE_ORDER


class JsonlReportWriter:
    """Контекстный менеджер для аппенд-записи JSONL-отчёта."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh = None

    def __enter__(self) -> "JsonlReportWriter":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # 'w' — отчёт пересоздаётся на каждый запуск, чтобы не путать
        # старые цифры с новыми. Хочешь сравнивать — переименуй файл.
        self._fh = self._path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *_exc) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def write(self, report: SampleReport) -> None:
        if self._fh is None:
            raise RuntimeError("JsonlReportWriter used outside of `with` block")
        self._fh.write(json.dumps(report.to_dict(), ensure_ascii=False))
        self._fh.write("\n")
        self._fh.flush()


# ---- агрегация по всем сэмплам -------------------------------------------

def aggregate(reports: Iterable[SampleReport]) -> dict:
    """Усреднить метрики по этапам.

    Возвращаемая структура:

        {
          "n_samples": N,
          "stages": {
            "rag":       {ok_rate, mean_precision, mean_recall, mean_f1, ...},
            "anchor":    {ok_rate, anchor_in_required_rate, anchor_fallback_rate, ...},
            "neighbors": {...},
            "prune":     {...}
          }
        }

    Считаем средние **только по этапам, которые реально выполнились** (для
    каждого сэмпла отдельно). Это правильнее, чем подмешивать нули за
    «не дошёл до этапа» — иначе среднее зависело бы от --until.
    """
    reports_list = list(reports)
    if not reports_list:
        return {"n_samples": 0, "stages": {}}

    by_stage: dict[str, list[dict]] = {}
    for r in reports_list:
        for m in r.stage_metrics:
            by_stage.setdefault(m.stage, []).append(m.to_dict())

    out: dict = {"n_samples": len(reports_list), "stages": {}}
    for stage_name in STAGE_ORDER:
        rows = by_stage.get(stage_name.value)
        if not rows:
            continue
        ok_rows = [r for r in rows if r["ok"]]
        agg: dict = {
            "executed": len(rows),
            "ok": len(ok_rows),
            "ok_rate": round(len(ok_rows) / len(rows), 4),
        }
        if ok_rows:
            for key in (
                "size",
                "coverage_required",
                "coverage_useful",
                "precision",
                "recall",
                "f1",
                "true_positive",
                "predicted",
                "gold_all"
            ):
                values = [r[key] for r in ok_rows]
                agg[f"mean_{key}"] = round(statistics.fmean(values), 4)
            # «На скольких сэмплах этап вытащил хотя бы один required-класс».
            # Абсолютное число важно само по себе (даёт грубый счёт «удач»),
            # доля — для сравнения этапов между собой.
            n_hit = sum(1 for r in ok_rows if r.get("hit_any_required"))
            micro_precision = sum([r["true_positive"] for r in ok_rows]) / sum([r["predicted"] for r in ok_rows])
            micro_recall = sum([r["true_positive"] for r in ok_rows]) / sum([r["gold_all"] for r in ok_rows])
            agg["n_with_any_required"] = n_hit
            agg["any_required_rate"] = round(n_hit / len(ok_rows), 4)
            agg["micro_precision"] = micro_precision
            agg["micro_recall"] = micro_recall
        # Спец-агрегации этапа anchor.
        if stage_name.value == "anchor" and ok_rows:
            agg["anchor_in_required_rate"] = round(
                sum(1 for r in ok_rows if r.get("anchor_in_required")) / len(ok_rows),
                4,
            )
            agg["anchor_fallback_rate"] = round(
                sum(1 for r in ok_rows if r.get("anchor_fallback")) / len(ok_rows),
                4,
            )
        out["stages"][stage_name.value] = agg
    return out
