"""Общий контракт «вывод одного этапа».

Все четыре этапа возвращают `StageOutcome`. Это:

* делает интерфейс между этапами одинаковым (legible: видно, что один и тот
  же объект течёт по конвейеру);
* даёт ровно один тип, который умеет считать `metrics.py` для оценки
  качества этапа против ground-truth;
* позволяет писать поэтапный JSONL-отчёт без дополнительных адаптеров.

`node_ids` — это «классы, которые этап считает кандидатами на ответ».
Именно по нему считаются coverage/precision/recall на каждом этапе:

    * stage 1 retrieve : top-K кандидатов от RAG
    * stage 2 anchor   : {anchor}                 (одиночный класс)
    * stage 3 expand   : {anchor} ∪ соседи        (1-hop окрестность)
    * stage 4 prune    : required ∪ useful        (выход LLM-пруна)

`payload` хранит детали, специфичные для конкретного этапа (баллы RAG,
формулировку причины выбора anchor, признак усечения подграфа,
разделение required/useful). Это «сырьё» для дебага и для следующего
этапа; метрики смотрят только на `node_ids`.

`info` — короткие диагностические числа (длительность, размер ответа LLM,
число входных/выходных токенов и т.п.). Сюда же попадают пометки об
аборте этапа (`aborted=...`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StageName(str, Enum):
    """Имена этапов — единая точка истины (используются и в CLI флаге `--until`)."""

    RETRIEVE = "rag"        # этап 1: RAG-кандидаты
    ANCHOR = "anchor"       # этап 2: выбор anchor
    NEIGHBORS = "neighbors" # этап 3: 1-hop окрестность
    PRUNE = "prune"         # этап 4: LLM-прунинг


# Порядок выполнения. Используем для срезов «выполнить до этого этапа».
STAGE_ORDER: tuple[StageName, ...] = (
    StageName.RETRIEVE,
    StageName.ANCHOR,
    StageName.NEIGHBORS,
    StageName.PRUNE,
)


@dataclass
class StageOutcome:
    """Стандартный результат одного этапа.

    Атрибуты:
        stage     : какой это этап (из StageName).
        node_ids  : множество node_id, которое этап «оставляет в кадре».
                    Именно по нему считаются метрики качества этапа.
        payload   : данные, специфичные для этапа (см. модуль каждого
                    этапа за схемой). JSON-сериализуем.
        info      : плоский словарь диагностических значений (длительности,
                    счётчики, флаг аборта). JSON-сериализуем.
        aborted   : пустая строка == всё хорошо. Иначе — короткий код,
                    почему этап не дошёл до результата (cм. поле info).
    """

    stage: StageName
    node_ids: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    aborted: str = ""

    def is_ok(self) -> bool:
        """Этап завершился штатно (не аборт)."""
        return not self.aborted

    def to_dict(self) -> dict[str, Any]:
        """JSON-вид для дебаг-отчёта."""
        return {
            "stage": self.stage.value,
            "node_ids": list(self.node_ids),
            "payload": self.payload,
            "info": self.info,
            "aborted": self.aborted,
        }
