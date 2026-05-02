"""Загрузка эталонных меток (ground-truth) — отдельным каналом от пайплайна.

Зачем отдельный модуль: в `data/dataset_queries.csv` остались только поля
`(repo, query)` — это то, что мы реально подаём в пайплайн (нет утечки).
А `data/dataset.csv` содержит все аннотации. Этот модуль матчит slim-строки
с полным датасетом по ключу `(repo, query)` и отдаёт `GoldLabels`.

Pipeline сам про эти метки ничего не знает: они приходят в пайплайн только
после того, как этап завершился, — внутри `metrics.py`/`run.py`.

Если по `(repo, query)` нашлось несколько строк (один и тот же запрос мог
быть аннотирован несколько раз), берём первую и предупреждаем — в рамках
текущего датасета это редкий случай, но честнее зафиксировать его.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.eval.annotations import AnnotationSample, load_dataset


@dataclass(frozen=True)
class GoldLabels:
    """Эталон для одного (repo, query)."""
    repo: str
    query: str
    central_node: str
    required: frozenset[str]   # node_id'ы помеченные как "required"
    useful: frozenset[str]     # node_id'ы помеченные как "useful"
    sample_id: str = ""        # для прослеживаемости в отчёте


class GoldTruthIndex:
    """Быстрый поиск GoldLabels по (repo, query)."""

    def __init__(self, samples: list[AnnotationSample]) -> None:
        self._by_key: dict[tuple[str, str], GoldLabels] = {}
        self._duplicates: list[tuple[str, str]] = []
        for s in samples:
            key = (s.repo, s.query)
            req = frozenset(
                nid for nid, kind in s.annotations.items() if kind == "required"
            )
            use = frozenset(
                nid for nid, kind in s.annotations.items() if kind == "useful"
            )
            labels = GoldLabels(
                repo=s.repo,
                query=s.query,
                central_node=s.central_node,
                required=req,
                useful=use,
                sample_id=s.sample_id,
            )
            if key in self._by_key:
                self._duplicates.append(key)
                continue  # сохраняем первый, остальные игнорируем
            self._by_key[key] = labels

    @classmethod
    def from_csv(cls, path: str | Path) -> "GoldTruthIndex":
        return cls(load_dataset(path))

    def lookup(self, repo: str, query: str) -> GoldLabels | None:
        return self._by_key.get((repo, query))

    @property
    def duplicates(self) -> list[tuple[str, str]]:
        """Ключи, по которым было несколько эталонных строк (мы оставили первую)."""
        return list(self._duplicates)

    def __len__(self) -> int:
        return len(self._by_key)
