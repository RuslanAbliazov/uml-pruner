"""Этап 1 — RAG: подобрать top-K кандидатов под запрос.

Что делает:

1. Берёт заранее построенный эмбеддинг-индекс диаграммы из
   ``data/embeddings/<diagram_stem>/`` (создаётся ``scripts/build_index.py``).
2. Кодирует запрос той же моделью эмбеддингов.
3. Возвращает топ-K node_id по косинусной близости.

Почему класс, а не функция: индекс и сам энкодер дороги при первом
обращении (загрузка модели, mmap векторов). Класс владеет ими и кэширует
по `diagram_stem`, чтобы прогон по 10 сэмплам из одного репо не
переинициализировал ничего повторно.

Если индекс отсутствует или устарел — возвращаем «аборт-вид» StageOutcome
с пустым `node_ids` и понятным `aborted` кодом. Pipeline в этом случае
не идёт дальше; в дебаг-отчёте это будет видно сразу.
"""

from __future__ import annotations

from typing import Any, Optional

from src.approaches.anchor_neighbors.stage_outputs import StageName, StageOutcome
from src.rag.cache import (
    EmbeddingCacheEntry,
    compute_diagram_hash,
    is_valid,
    load_cache,
)
from src.rag.encoder import EncoderConfig, LocalEncoder
from src.rag.retriever import retrieve_top_k


def _short_name(node_id: str) -> str:
    """Короткое имя класса (последний сегмент полного node_id)."""
    return node_id.rsplit(".", 1)[-1] if "." in node_id else node_id


class CandidateRetriever:
    """Хранит энкодер и кэш индексов между сэмплами одного запуска."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        batch_size: int,
        cache_dir: str,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._cache_dir = cache_dir
        # Ленивая инициализация — модель грузится только при первом запросе.
        self._encoder: Optional[LocalEncoder] = None
        self._index_by_stem: dict[str, EmbeddingCacheEntry] = {}

    # ---- публичный вход ------------------------------------------------

    def run(
        self,
        *,
        query: str,
        diagram_stem: str,
        nodes: list[dict[str, Any]],
        top_k: int,
    ) -> StageOutcome:
        """Главная точка этапа: вернуть StageOutcome с RAG-кандидатами."""
        index = self._load_index(diagram_stem, nodes)
        if index is None:
            return StageOutcome(
                stage=StageName.RETRIEVE,
                aborted="no_index",
                info={
                    "diagram_stem": diagram_stem,
                    "hint": (
                        "запусти `python scripts/build_index.py` "
                        "для этой диаграммы"
                    ),
                },
            )

        encoder = self._get_encoder()
        hits = retrieve_top_k(query, index, encoder, top_k=top_k)

        # `hits` упорядочен по убыванию score. Нам важно сохранить именно
        # этот порядок: следующий этап (выбор anchor) подаёт их в LLM
        # «highest first», и мы же используем top-1 как fallback.
        node_by_id = {n["node_id"]: n for n in nodes if n.get("node_id")}
        candidates: list[dict[str, Any]] = []
        for h in hits:
            node = node_by_id.get(h.node_id)
            if node is None:
                # Разошлись индекс и текущая нормализация — пропускаем тихо;
                # это не аборт этапа, мы просто отдадим, что валидно.
                continue
            candidates.append(
                {
                    "node_id": h.node_id,
                    "name": node.get("name") or _short_name(h.node_id),
                    "type": node.get("type", "class"),
                    "score": round(float(h.score), 6),
                }
            )

        return StageOutcome(
            stage=StageName.RETRIEVE,
            node_ids=[c["node_id"] for c in candidates],
            payload={"candidates": candidates},
            info={
                "diagram_stem": diagram_stem,
                "top_k_requested": top_k,
                "top_k_returned": len(candidates),
            },
        )

    # ---- внутренности (lazy init индекса/энкодера) ---------------------

    def _get_encoder(self) -> LocalEncoder:
        if self._encoder is None:
            self._encoder = LocalEncoder(
                EncoderConfig(
                    model_name=self._model_name,
                    device=self._device,
                    batch_size=self._batch_size,
                )
            )
        return self._encoder

    def _load_index(
        self, diagram_stem: str, nodes: list[dict[str, Any]]
    ) -> Optional[EmbeddingCacheEntry]:
        """Грузим индекс с диска один раз на diagram_stem за весь прогон.

        `is_valid` проверяет, что (а) индекс собран той же моделью эмбеддингов,
        что задана в YAML, и (б) хэш набора узлов совпадает с тем, что был на
        момент построения индекса. Если что-то расходится — возвращаем None,
        этап ляжет с `aborted=stale_index/no_index`.
        """
        cached = self._index_by_stem.get(diagram_stem)
        if cached is not None:
            return cached

        entry = load_cache(self._cache_dir, diagram_stem)
        if entry is None:
            return None
        if not is_valid(
            entry,
            expected_model=self._model_name,
            expected_diagram_hash=compute_diagram_hash(nodes),
        ):
            return None
        self._index_by_stem[diagram_stem] = entry
        return entry
