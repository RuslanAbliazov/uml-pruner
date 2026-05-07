"""Метрики качества каждого этапа подхода ``anchor_neighbors``.

Идея: каждый этап выдаёт `StageOutcome.node_ids`. Мы сравниваем это
множество с эталонными required/useful классами из ground-truth и для
каждого этапа считаем:

* coverage_required  — какая доля эталонных REQUIRED попала в выход этапа
* coverage_useful    — какая доля эталонных USEFUL  попала в выход этапа
* precision          — |выход ∩ (required∪useful)| / |выход|
* recall             — |выход ∩ (required∪useful)| / |required∪useful|
* f1                 — гармоническое среднее precision/recall
* size               — |выход| (для интерпретации precision)

Зачем разделять coverage по required и useful: на этапе RAG мы не ждём
полного recall'a, но коробочно хотим видеть, что хотя бы required-классы
*перекрываются* с топ-K — это идеальный сигнал, что pipeline начал хорошо.

Особенность этапа 2 (anchor): множество выходов размера 1 (anchor)
делает обычные метрики малоинформативными, поэтому добавляем
`anchor_in_required` / `anchor_in_useful` / `anchor_in_gt` булевы флаги.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.approaches.anchor_neighbors.ground_truth import GoldLabels
from src.approaches.anchor_neighbors.stage_outputs import (
    STAGE_ORDER,
    StageName,
    StageOutcome,
)


# ---- базовые числовые метрики --------------------------------------------

def _safe_div(num: int, den: int) -> float:
    return num / den if den else 0.0


@dataclass
class StageMetrics:
    """Одна строка метрик для одного этапа.

    Все поля — JSON-сериализуемые скаляры. Отдельные поля для каждого
    показателя удобнее, чем nested dict, для скриптов анализа потом.
    """
    stage: str
    ok: bool                   # этап завершился без аборта
    aborted: str = ""          # если ok=False — короткий код причины
    size: int = 0              # |вывод|
    n_gold_required: int = 0
    n_gold_useful: int = 0
    coverage_required: float = 0.0
    coverage_useful: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    true_positive: int = 0
    predicted: int = 0
    gold_all: int = 0
    # «Удалось ли вытащить хотя бы один required-класс на этом этапе?»
    # Полезно как грубый бинарный индикатор: даже если recall маленький
    # из-за множества required, важно понимать, на скольких сэмплах
    # этап вообще промахнулся мимо ВСЕХ required.
    hit_any_required: bool = False
    # Только для этапа anchor:
    anchor: str = ""
    anchor_in_required: bool = False
    anchor_in_useful: bool = False
    anchor_fallback: bool = False  # True, если LLM ошибся и сработал top-1 fallback

    def to_dict(self) -> dict[str, Any]:
        # Удобный формат для JSONL: пропускаем поля anchor_*, если этап != anchor.
        d = {
            "stage": self.stage,
            "ok": self.ok,
            "aborted": self.aborted,
            "size": self.size,
            "n_gold_required": self.n_gold_required,
            "n_gold_useful": self.n_gold_useful,
            "coverage_required": round(self.coverage_required, 4),
            "coverage_useful": round(self.coverage_useful, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positive": self.true_positive,
            "predicted": self.predicted,
            "gold_all": self.gold_all,
            "hit_any_required": self.hit_any_required,
        }
        if self.stage == StageName.ANCHOR.value:
            d.update({
                "anchor": self.anchor,
                "anchor_in_required": self.anchor_in_required,
                "anchor_in_useful": self.anchor_in_useful,
                "anchor_fallback": self.anchor_fallback,
            })
        return d


def _compute_core(predicted: set[str], gold: GoldLabels) -> dict[str, float]:
    """Считает coverage_required/coverage_useful/precision/recall/f1.

    `predicted` — множество node_id, которое выдал этап.
    Объединение required и useful трактуется как «положительные» примеры
    (т.е. precision/recall/f1 — против него).
    """
    gold_all = gold.required | gold.useful
    tp = predicted & gold_all
    precision = _safe_div(len(tp), len(predicted))
    recall = _safe_div(len(tp), len(gold_all))
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return {
        "coverage_required": _safe_div(len(predicted & gold.required), len(gold.required)),
        "coverage_useful":   _safe_div(len(predicted & gold.useful),   len(gold.useful)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": len(tp),
        "predicted": len(predicted),
        "gold_all": len(gold_all)
    }


# ---- метрики на конкретный этап ------------------------------------------

def metrics_for_stage(outcome: StageOutcome, gold: GoldLabels) -> StageMetrics:
    """Сосчитать `StageMetrics` для одного этапа."""
    base = StageMetrics(
        stage=outcome.stage.value,
        ok=outcome.is_ok(),
        aborted=outcome.aborted,
        size=len(outcome.node_ids),
        n_gold_required=len(gold.required),
        n_gold_useful=len(gold.useful),
    )
    if not outcome.is_ok():
        return base  # Аборт — все метрики остаются нулями, причина в `aborted`.

    predicted = set(outcome.node_ids)
    core = _compute_core(predicted, gold)
    base.coverage_required = core["coverage_required"]
    base.coverage_useful = core["coverage_useful"]
    base.precision = core["precision"]
    base.recall = core["recall"]
    base.f1 = core["f1"]
    base.true_positive = core["true_positive"]
    base.predicted = core["predicted"]
    base.gold_all = core["gold_all"]
    # Хотя бы один required-класс попал в выход этапа.
    # Если эталонных required нет вовсе — считаем флаг ложным
    # (нечего ловить → нечего фиксировать как «успех»).
    base.hit_any_required = bool(gold.required and (predicted & gold.required))

    # Спец-поля для anchor-этапа: показывают, попал ли выбранный anchor
    # в эталонные required/useful, и был ли это fallback на top-1.
    if outcome.stage is StageName.ANCHOR and outcome.node_ids:
        anchor = outcome.node_ids[0]
        base.anchor = anchor
        base.anchor_in_required = anchor in gold.required
        base.anchor_in_useful = anchor in gold.useful
        base.anchor_fallback = bool(outcome.payload.get("fallback"))

    return base


# ---- метрики на весь пайплайн (по всем выполненным этапам) ---------------

@dataclass
class SampleReport:
    """Поэтапный отчёт по одному сэмплу — то, что пишется в JSONL."""
    sample_id: str
    repo: str
    query: str
    stage_metrics: list[StageMetrics] = field(default_factory=list)
    stage_payloads: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "repo": self.repo,
            "query": self.query,
            "metrics": [m.to_dict() for m in self.stage_metrics],
            "stages": self.stage_payloads,
        }


def build_sample_report(
    *,
    sample_id: str,
    repo: str,
    query: str,
    stages: dict[StageName, StageOutcome],
    gold: GoldLabels,
) -> SampleReport:
    """Собрать поэтапный отчёт для одного сэмпла.

    Идём по фиксированному порядку этапов — пропускаем те, что не
    выполнялись (например, при `--until rag` будет ровно один этап).
    """
    rows = [
        metrics_for_stage(stages[name], gold)
        for name in STAGE_ORDER
        if name in stages
    ]
    payloads = {
        name.value: stages[name].to_dict()
        for name in STAGE_ORDER
        if name in stages
    }
    return SampleReport(
        sample_id=sample_id,
        repo=repo,
        query=query,
        stage_metrics=rows,
        stage_payloads=payloads,
    )
