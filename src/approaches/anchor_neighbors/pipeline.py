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

from src.approaches._common.compressor import filter_subgraph
from src.approaches.anchor_neighbors import (
    stage1_retrieve,
    stage2_select_anchor,
    stage3_expand_neighbors,
    stage4_prune,
)
from src.approaches.anchor_neighbors.settings import AnchorNeighborsSettings
from src.approaches.anchor_neighbors.stage_outputs import (
    STAGE_ORDER,
    StageName,
    StageOutcome,
)
from src.core.types import ApproachInputs, ApproachResult
from src.eval.annotations import diagram_filename_for_repo
from src.llm.client import LLMClient

NAME = "anchor_neighbors"


@dataclass
class PipelineOutcome:
    """Полный результат одного запуска: и сырьё для оценки, и финал."""
    stages: dict[StageName, StageOutcome] = field(default_factory=dict)
    result: ApproachResult = field(
        default_factory=lambda: ApproachResult(approach=NAME)
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
        s1 = self._retriever.run(
            query=inputs.query,
            diagram_stem=diagram_stem,
            nodes=nodes,
            top_k=self._settings.pipeline.n_candidates,
        )
        stages[StageName.RETRIEVE] = s1
        if not s1.is_ok() or until == StageName.RETRIEVE:
            return _build_outcome(stages, inputs, node_by_id, edges)

        # ---- этап 2 ---------------------------------------------------
        s2 = await stage2_select_anchor.select_anchor(
            query=inputs.query,
            candidates=s1.payload["candidates"],
            node_by_id=node_by_id,
            edges=edges,
            llm=self._llm,
        )
        stages[StageName.ANCHOR] = s2
        if not s2.is_ok() or until == StageName.ANCHOR:
            return _build_outcome(stages, inputs, node_by_id, edges)

        anchor = s2.payload["anchor"]

        # ---- этап 3 ---------------------------------------------------
        s3 = stage3_expand_neighbors.expand_neighbors(
            anchor=anchor,
            nodes=nodes,
            edges=edges,
            node_by_id=node_by_id,
            cap=self._settings.pipeline.max_subgraph_nodes,
        )
        stages[StageName.NEIGHBORS] = s3
        if not s3.is_ok() or until == StageName.NEIGHBORS:
            return _build_outcome(stages, inputs, node_by_id, edges)

        # ---- этап 4 ---------------------------------------------------
        s4 = await stage4_prune.prune_subgraph(
            query=inputs.query,
            anchor=anchor,
            sub_nodes=s3.payload["sub_nodes"],
            sub_edges=s3.payload["sub_edges"],
            llm=self._llm,
        )
        stages[StageName.PRUNE] = s4
        return _build_outcome(stages, inputs, node_by_id, edges)

    # ---- совместимость с интерфейсом ApproachRunner -------------------

    async def run(self, inputs: ApproachInputs) -> ApproachResult:
        """Минимальный интерфейс для реестра подходов: всегда до prune."""
        outcome = await self.run_with_stages(inputs, until=StageName.PRUNE)
        return outcome.result

    async def aclose(self) -> None:
        """Долгоживущих ресурсов нет; метод нужен только из-за протокола."""
        return None


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
