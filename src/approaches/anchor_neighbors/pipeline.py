"""Оркестратор подхода ``anchor_neighbors``.

Соединяет 4 этапа: retrieve → anchor → neighbors → prune.

Главное API: `AnchorNeighborsPipeline.run(inputs, until)`. Возвращает
объект `PipelineOutcome`, который держит:

* `stages: dict[StageName, StageOutcome]` — результат каждого выполненного
  этапа (для дебаг-отчёта и оценки по этапам);
* `result: ApproachResult` — стандартный продакшн-вид результата
  (узлы/рёбра отфильтрованного подграфа + required/useful + metadata).

Для совместимости с реестром (`src.approaches.__init__`) реализуем
протокол `ApproachRunner`: метод `run(inputs)` без `until`. Он просто
вызывает `run(inputs, until=PRUNE)` и возвращает только `result`.

Гарантии этого файла:
* НИКАКОГО ground-truth внутри. Pipeline видит только `(query, diagram)`.
  Метрики считаются СНАРУЖИ, в `metrics.py`, на основе `stages`.
* Никакого логирования. Все диагностические данные — в `StageOutcome.info`,
  они дойдут до пользователя через дебаг-отчёт.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import json
from pathlib import Path

from src.approaches._common.compressor import filter_subgraph
from src.approaches.anchor_neighbors import (
    stage1_retrieve,
    stage2_rerank_anchor,
    stage2_select_anchor,
    stage3_expand_neighbors,
    stage4_prune,
)
from src.approaches.anchor_neighbors.error_logger import ErrorLogger
from src.approaches.anchor_neighbors.llm_trace import LLMTracer
from src.approaches.anchor_neighbors.settings import AnchorNeighborsSettings
from src.approaches.anchor_neighbors.stage_outputs import (
    STAGE_ORDER,
    StageName,
    StageOutcome,
)
from src.core.types import ApproachInputs, ApproachResult
from src.eval.annotations import diagram_filename_for_repo
from src.llm.client import LLMClient
from src.rag.reranker import LocalReranker, RerankerConfig

NAME = "anchor_neighbors"


@dataclass
class PipelineOutcome:
    """Полный результат одного запуска: и сырьё для оценки, и финал."""
    stages: dict[StageName, StageOutcome] = field(default_factory=dict)
    result: ApproachResult = field(
        default_factory=lambda: ApproachResult(approach=NAME)
    )


def _save_stage2_anchors(
    anchors: list[str],
    query: str,
    inputs: ApproachInputs,
) -> None:
    """Сохранить список anchor-классов после этапа 2."""
    out_dir = Path("data/stage2_anchors")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_repo = inputs.repo.replace("/", "_") if inputs.repo else "unknown"
    sample_id = inputs.sample_id or "nosample"
    fname = f"{safe_repo}__{sample_id}.json"
    with (out_dir / fname).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "anchors": anchors,
                "query": query,
                "sample_id": inputs.sample_id,
                "repo": inputs.repo,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def _save_stage3_graph(
    sub_nodes: list[dict[str, Any]],
    sub_edges: list[dict[str, Any]],
    inputs: ApproachInputs,
) -> None:
    out_dir = Path("data/stage3_graphs")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_repo = inputs.repo.replace("/", "_") if inputs.repo else "unknown"
    sample_id = inputs.sample_id or "nosample"
    fname = f"{safe_repo}__{sample_id}.json"
    with (out_dir / fname).open("w", encoding="utf-8") as f:
        json.dump(
            {"nodes": sub_nodes, "edges": sub_edges},
            f,
            ensure_ascii=False,
            indent=2,
            default=str,   # <-- спасает от несериализуемых типов
        )


def _save_stage3_prompt_data(
    query: str,
    sub_nodes: list[dict[str, Any]],
    sub_edges: list[dict[str, Any]],
    inputs: ApproachInputs,
) -> None:
    """Сохранить данные для промпта этапа 4, чтобы можно было пропустить этапы 1-3."""
    out_dir = Path("data/stage3_prompts")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_repo = inputs.repo.replace("/", "_") if inputs.repo else "unknown"
    sample_id = inputs.sample_id or "nosample"
    fname = f"{safe_repo}__{sample_id}.json"
    with (out_dir / fname).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "query": query,
                "sub_nodes": sub_nodes,
                "sub_edges": sub_edges,
                "sample_id": inputs.sample_id,
                "repo": inputs.repo,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


class AnchorNeighborsPipeline:
    """Реализация подхода #2 в виде явного 4-этапного пайплайна."""

    name = NAME

    def __init__(
        self,
        settings: AnchorNeighborsSettings,
        llm: LLMClient,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._retriever = stage1_retrieve.CandidateRetriever(
            model_name=settings.retriever.model_name,
            device=settings.retriever.device,
            batch_size=settings.retriever.batch_size,
            cache_dir=settings.retriever.cache_dir,
        )
        # Трейсер LLM-вызовов: пишет последний request/response на (этап, sample_id).
        # Папку создаст сам при первом обращении; если sample_id у inputs
        # пустой — записи просто не будет (см. stage2/stage4).
        self._tracer = LLMTracer(settings.pipeline.llm_traces_dir)
        # Логгер ошибок: пишет полный traceback + контекст при любой ошибке
        error_log_dir = Path("data/errors/anchor_neighbors") / settings.pipeline.anchor_selector
        self._error_logger = ErrorLogger(error_log_dir)
        # Reranker инстанцируется лениво — только если выбрана ветка
        # `anchor_selector == "reranker"`. В режиме `"llm"` модель никогда
        # не загружается, и тяжёлые torch-зависимости не нужны.
        self._reranker: LocalReranker | None = None
        if settings.pipeline.anchor_selector == "reranker":
            assert settings.reranker is not None  # гарантировано load_settings()
            self._reranker = LocalReranker(
                RerankerConfig(
                    model_name=settings.reranker.model_name,
                    device=settings.reranker.device,
                    batch_size=settings.reranker.batch_size,
                    max_seq_length=settings.reranker.max_seq_length,
                )
            )

    @property
    def settings(self) -> AnchorNeighborsSettings:
        return self._settings

    # ---- основной публичный интерфейс ---------------------------------

    async def run_with_stages(
        self,
        inputs: ApproachInputs,
        until: StageName = StageName.PRUNE,
    ) -> PipelineOutcome:
        """Прогнать пайплайн до указанного этапа включительно.

        Возвращает `PipelineOutcome` со всеми промежуточными результатами.
        Если этап завершился с `aborted` — следующие этапы не запускаются;
        в `stages` останутся только дошедшие.
        """
        stages: dict[StageName, StageOutcome] = {}

        # Подготовка: индексируем узлы по id; вычисляем имя файла индекса.
        nodes = inputs.nodes
        edges = inputs.edges
        node_by_id = {n["node_id"]: n for n in nodes if n.get("node_id")}
        diagram_stem = _diagram_stem(inputs.repo)

        # ---- этап 1 ---------------------------------------------------
        try:
            s1 = self._retriever.run(
                query=inputs.query,
                diagram_stem=diagram_stem,
                nodes=nodes,
                top_k=self._settings.pipeline.n_candidates,
            )
        except Exception as e:
            self._error_logger.log_stage_error(
                stage_name="retrieve",
                sample_id=inputs.sample_id,
                exception=e,
                context={
                    "query": inputs.query,
                    "repo": inputs.repo,
                    "diagram_stem": diagram_stem,
                    "nodes_count": len(nodes),
                    "top_k": self._settings.pipeline.n_candidates,
                },
            )
            raise  # Пробрасываем дальше, чтобы run.py тоже увидел
        
        stages[StageName.RETRIEVE] = s1
        
        # Если этап завершился с aborted, логируем детали
        if not s1.is_ok():
            self._error_logger.log_stage_error(
                stage_name="retrieve",
                sample_id=inputs.sample_id,
                exception=Exception(f"Stage aborted: {s1.aborted}"),
                context={
                    "query": inputs.query,
                    "repo": inputs.repo,
                    "aborted_reason": s1.aborted,
                    "stage_info": s1.info,
                },
            )
        
        if not s1.is_ok() or until == StageName.RETRIEVE:
            return _build_outcome(stages, inputs, node_by_id, edges)

        # ---- этап 2 ---------------------------------------------------
        # Развилка по `anchor_selector`. Обе ветки возвращают одинаковый
        # `StageOutcome(stage=ANCHOR, ...)` с полями `payload.anchors` (top-N)
        # и `payload.anchor` (top-1 — для обратной совместимости).
        n_anchors = self._settings.pipeline.n_anchors
        try:
            if self._settings.pipeline.anchor_selector == "reranker":
                assert self._reranker is not None  # гарантировано __init__
                s2 = await stage2_rerank_anchor.select_anchor_via_reranker(
                    query=inputs.query,
                    candidates=s1.payload["candidates"],
                    node_by_id=node_by_id,
                    edges=edges,
                    reranker=self._reranker,
                    n_anchors=n_anchors,
                )
            else:
                s2 = await stage2_select_anchor.select_anchor(
                    query=inputs.query,
                    candidates=s1.payload["candidates"],
                    node_by_id=node_by_id,
                    edges=edges,
                    llm=self._llm,
                    n_anchors=n_anchors,
                    tracer=self._tracer,
                    sample_id=inputs.sample_id,
                )
        except Exception as e:
            self._error_logger.log_stage_error(
                stage_name="anchor",
                sample_id=inputs.sample_id,
                exception=e,
                context={
                    "query": inputs.query,
                    "repo": inputs.repo,
                    "candidates_count": len(s1.payload.get("candidates", [])),
                    "n_anchors": n_anchors,
                    "anchor_selector": self._settings.pipeline.anchor_selector,
                },
            )
            raise
        
        stages[StageName.ANCHOR] = s2
        
        if not s2.is_ok():
            self._error_logger.log_stage_error(
                stage_name="anchor",
                sample_id=inputs.sample_id,
                exception=Exception(f"Stage aborted: {s2.aborted}"),
                context={
                    "query": inputs.query,
                    "repo": inputs.repo,
                    "aborted_reason": s2.aborted,
                    "stage_info": s2.info,
                },
            )
        
        if not s2.is_ok() or until == StageName.ANCHOR:
            return _build_outcome(stages, inputs, node_by_id, edges)

        # `anchors` — основной список; если по какой-то причине его нет в
        # payload (например, кастомный stage2), берём top-1 как раньше.
        anchors: list[str] = list(
            s2.payload.get("anchors") or [s2.payload["anchor"]]
        )
        
        # Сохраняем anchor-классы для возможности пропуска этапов 1-2
        if s2.is_ok():
            _save_stage2_anchors(
                anchors=anchors,
                query=inputs.query,
                inputs=inputs,
            )

        # ---- этап 3 ---------------------------------------------------
        try:
            s3 = stage3_expand_neighbors.expand_neighbors(
                anchors=anchors,
                nodes=nodes,
                edges=edges,
                node_by_id=node_by_id,
                cap=self._settings.pipeline.max_subgraph_nodes,
            )
        except Exception as e:
            self._error_logger.log_stage_error(
                stage_name="neighbors",
                sample_id=inputs.sample_id,
                exception=e,
                context={
                    "query": inputs.query,
                    "repo": inputs.repo,
                    "anchors": anchors,
                    "nodes_count": len(nodes),
                    "edges_count": len(edges),
                    "cap": self._settings.pipeline.max_subgraph_nodes,
                    # Сохраняем пример ребра для диагностики структуры
                    "edge_sample": edges[0] if edges else None,
                    "node_sample": nodes[0] if nodes else None,
                },
            )
            raise
        
        stages[StageName.NEIGHBORS] = s3
        
        if not s3.is_ok():
            self._error_logger.log_stage_error(
                stage_name="neighbors",
                sample_id=inputs.sample_id,
                exception=Exception(f"Stage aborted: {s3.aborted}"),
                context={
                    "query": inputs.query,
                    "repo": inputs.repo,
                    "anchors": anchors,
                    "aborted_reason": s3.aborted,
                    "stage_info": s3.info,
                },
            )
        
        # Сохраняем данные для этапа 4 (промпт), чтобы можно было пропустить этапы 1-3
        if s3.is_ok():
            _save_stage3_prompt_data(
                query=inputs.query,
                sub_nodes=s3.payload["sub_nodes"],
                sub_edges=s3.payload["sub_edges"],
                inputs=inputs,
            )
            _save_stage3_graph(
                s3.payload["sub_nodes"],
                s3.payload["sub_edges"],
                inputs=inputs,
            )
        
        if not s3.is_ok() or until == StageName.NEIGHBORS:
            return _build_outcome(stages, inputs, node_by_id, edges)

        # ---- этап 4 ---------------------------------------------------
        try:
            s4 = await stage4_prune.prune_subgraph(
                query=inputs.query,
                sub_nodes=s3.payload["sub_nodes"],
                sub_edges=s3.payload["sub_edges"],
                llm=self._llm,
                prune_steps=self._settings.pipeline.prune_steps,
                tracer=self._tracer,
                sample_id=inputs.sample_id,
            )
        except Exception as e:
            self._error_logger.log_stage_error(
                stage_name="prune",
                sample_id=inputs.sample_id,
                exception=e,
                context={
                    "query": inputs.query,
                    "repo": inputs.repo,
                    "sub_nodes_count": len(s3.payload["sub_nodes"]),
                    "sub_edges_count": len(s3.payload["sub_edges"]),
                    "prune_steps": self._settings.pipeline.prune_steps,
                },
            )
            raise
        
        stages[StageName.PRUNE] = s4
        
        if not s4.is_ok():
            self._error_logger.log_stage_error(
                stage_name="prune",
                sample_id=inputs.sample_id,
                exception=Exception(f"Stage aborted: {s4.aborted}"),
                context={
                    "query": inputs.query,
                    "repo": inputs.repo,
                    "aborted_reason": s4.aborted,
                    "stage_info": s4.info,
                },
            )
        
        return _build_outcome(stages, inputs, node_by_id, edges)

    # ---- совместимость с интерфейсом ApproachRunner -------------------

    async def run(self, inputs: ApproachInputs) -> ApproachResult:
        """Минимальный интерфейс для реестра подходов: всегда до prune."""
        outcome = await self.run_with_stages(inputs, until=StageName.PRUNE)
        return outcome.result

    async def aclose(self) -> None:
        """Долгоживущих ресурсов нет; метод нужен только из-за протокола."""
        return None

    async def run_stage4_only(
        self,
        query: str,
        sub_nodes: list[dict[str, Any]],
        sub_edges: list[dict[str, Any]],
        sample_id: str = "",
    ) -> PipelineOutcome:
        """Запустить только этап 4 (prune) на готовых данных из stage3.
        
        Используется когда данные stage3 были сохранены ранее и нужно
        только выполнить LLM-прунинг без повторного выполнения этапов 1-3.
        """
        stages: dict[StageName, StageOutcome] = {}
        
        # Выполняем только этап 4
        s4 = await stage4_prune.prune_subgraph(
            query=query,
            sub_nodes=sub_nodes,
            sub_edges=sub_edges,
            llm=self._llm,
            prune_steps=self._settings.pipeline.prune_steps,
            tracer=self._tracer,
            sample_id=sample_id,
        )
        stages[StageName.PRUNE] = s4
        
        # Строим node_by_id из sub_nodes для _build_outcome
        node_by_id = {n["node_id"]: n for n in sub_nodes if n.get("node_id")}
        
        # Создаем минимальный ApproachInputs для _build_outcome
        inputs = ApproachInputs(
            query=query,
            diagram={"nodes": sub_nodes, "edges": sub_edges},
            sample_id=sample_id,
            repo="",  # Repo неизвестен при загрузке из stage3
        )
        
        return _build_outcome(stages, inputs, node_by_id, sub_edges)


# ---- сборка финального ApproachResult из последнего успешного этапа ----

def _build_outcome(
    stages: dict[StageName, StageOutcome],
    inputs: ApproachInputs,
    node_by_id: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> PipelineOutcome:
    """Из набора пройденных этапов слепить ApproachResult.

    Логика «последний успешный этап задаёт финальный подграф»:

    * Если выполнен prune — required/useful берутся из его payload,
      подграф = required ∪ useful.
    * Если до prune не дошли (например, при `--until neighbors`) —
      подграф = node_ids этапа neighbors, classification пустой.
    * Если не дошли и до neighbors — подграф пуст, метаданные
      сохраняют то, что успели увидеть.
    """
    metadata: dict[str, Any] = {
        "stages_executed": [s.value for s in stages],
    }

    # Прокинем краткие сводки по каждому этапу — удобно при ручном просмотре.
    for stage_name in STAGE_ORDER:
        outc = stages.get(stage_name)
        if outc is None:
            continue
        metadata[f"stage_{stage_name.value}"] = {
            "ok": outc.is_ok(),
            "aborted": outc.aborted,
            "size": len(outc.node_ids),
            **{k: v for k, v in outc.info.items() if _is_jsonable(v)},
        }

    # Определим финальный набор узлов/required/useful.
    required: list[str] = []
    useful: list[str] = []
    keep: set[str] = set()

    if (prune := stages.get(StageName.PRUNE)) and prune.is_ok():
        required = list(prune.payload.get("required", []))
        useful = list(prune.payload.get("useful", []))
        keep = set(required) | set(useful)
    elif (neigh := stages.get(StageName.NEIGHBORS)) and neigh.is_ok():
        keep = set(neigh.node_ids)
    elif (anch := stages.get(StageName.ANCHOR)) and anch.is_ok():
        keep = set(anch.node_ids)
    elif (retr := stages.get(StageName.RETRIEVE)) and retr.is_ok():
        keep = set(retr.node_ids)

    sub_nodes, sub_edges = filter_subgraph(
        list(node_by_id.values()), edges, keep
    )
    result = ApproachResult(
        approach=NAME,
        nodes=sub_nodes,
        edges=sub_edges,
        required_node_ids=sorted(required),
        useful_node_ids=sorted(useful),
        metadata=metadata,
    )
    return PipelineOutcome(stages=stages, result=result)


def _diagram_stem(repo: str) -> str:
    """Имя индекс-папки эмбеддингов под этот repo (без `.json`).

    Делегируем `diagram_filename_for_repo`, чтобы pipeline и benchmark
    смотрели в одну и ту же папку индексов даже после ручного маппинга
    отдельных репо."""
    fname = diagram_filename_for_repo(repo or "")
    return fname[:-5] if fname.endswith(".json") else fname


def _is_jsonable(v: Any) -> bool:
    """Простая защита от попадания в metadata несериализуемых объектов."""
    return isinstance(v, (str, int, float, bool, list, dict)) or v is None
