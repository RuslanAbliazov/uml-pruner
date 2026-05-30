"""Оркестратор подхода ``anchor_neighbors``.

Соединяет 4 этапа: retrieve → anchor → neighbors → prune.

Публичное API:
- `AnchorNeighborsPipeline.run_with_stages(inputs, until)` — прогнать до
  указанного этапа; вернуть `PipelineOutcome` со всеми промежуточными
  результатами (нужны для метрик и дебага).
- `AnchorNeighborsPipeline.run(inputs)` — тонкий адаптер для реестра
  подходов; всегда прогоняет до PRUNE.

При `until=NEIGHBORS` pipeline сохраняет `sub_nodes`/`sub_edges` в
`<outputs_dir>/stage3/<sample_id>.json` — чтобы потом можно было
запустить только LLM-прунинг через `--from-stage3`.

Гарантии:
- Никакого ground-truth внутри. Pipeline видит только `(query, diagram)`.
- Никакого логирования. Диагностика — в `StageOutcome.info`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    """Полный результат одного запуска."""
    stages: dict[StageName, StageOutcome] = field(default_factory=dict)
    result: ApproachResult = field(
        default_factory=lambda: ApproachResult(approach=NAME)
    )


class AnchorNeighborsPipeline:
    """Реализация подхода #2 в виде явного 4-этапного пайплайна."""

    name = NAME

    def __init__(self, settings: AnchorNeighborsSettings, llm: LLMClient) -> None:
        self._settings = settings
        self._llm = llm
        self._retriever = stage1_retrieve.CandidateRetriever(
            model_name=settings.retriever.model_name,
            device=settings.retriever.device,
            batch_size=settings.retriever.batch_size,
            cache_dir=settings.retriever.cache_dir,
        )
        self._tracer = LLMTracer(settings.pipeline.llm_traces_dir)
        self._error_logger = ErrorLogger(
            Path("data/errors/anchor_neighbors") / settings.pipeline.anchor_selector
        )
        self._reranker: LocalReranker | None = None
        if settings.pipeline.anchor_selector == "reranker":
            assert settings.reranker is not None
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

    async def run_with_stages(
        self,
        inputs: ApproachInputs,
        until: StageName = StageName.PRUNE,
    ) -> PipelineOutcome:
        """Прогнать пайплайн до указанного этапа включительно."""
        stages: dict[StageName, StageOutcome] = {}

        nodes = inputs.nodes
        edges = inputs.edges
        node_by_id = {n["node_id"]: n for n in nodes if n.get("node_id")}
        diagram_stem = _diagram_stem(inputs.repo)

        # ---- этап 1: RAG -----------------------------------------------
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
                context={"query": inputs.query, "repo": inputs.repo,
                         "diagram_stem": diagram_stem, "nodes_count": len(nodes),
                         "top_k": self._settings.pipeline.n_candidates},
            )
            raise

        stages[StageName.RETRIEVE] = s1
        if not s1.is_ok():
            self._error_logger.log_stage_error(
                stage_name="retrieve", sample_id=inputs.sample_id,
                exception=Exception(f"Stage aborted: {s1.aborted}"),
                context={"query": inputs.query, "repo": inputs.repo,
                         "aborted_reason": s1.aborted, "stage_info": s1.info},
            )
        if not s1.is_ok() or until == StageName.RETRIEVE:
            return _build_outcome(stages, inputs, node_by_id, edges)

        # ---- этап 2: выбор anchor --------------------------------------
        n_anchors = self._settings.pipeline.n_anchors
        try:
            if self._settings.pipeline.anchor_selector == "reranker":
                assert self._reranker is not None
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
                stage_name="anchor", sample_id=inputs.sample_id, exception=e,
                context={"query": inputs.query, "repo": inputs.repo,
                         "candidates_count": len(s1.payload.get("candidates", [])),
                         "n_anchors": n_anchors,
                         "anchor_selector": self._settings.pipeline.anchor_selector},
            )
            raise

        stages[StageName.ANCHOR] = s2
        if not s2.is_ok():
            self._error_logger.log_stage_error(
                stage_name="anchor", sample_id=inputs.sample_id,
                exception=Exception(f"Stage aborted: {s2.aborted}"),
                context={"query": inputs.query, "repo": inputs.repo,
                         "aborted_reason": s2.aborted, "stage_info": s2.info},
            )
        if not s2.is_ok() or until == StageName.ANCHOR:
            return _build_outcome(stages, inputs, node_by_id, edges)

        anchors: list[str] = list(
            s2.payload.get("anchors") or [s2.payload["anchor"]]
        )

        # ---- этап 3: расширение соседей --------------------------------
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
                stage_name="neighbors", sample_id=inputs.sample_id, exception=e,
                context={"query": inputs.query, "repo": inputs.repo,
                         "anchors": anchors, "nodes_count": len(nodes),
                         "edges_count": len(edges),
                         "cap": self._settings.pipeline.max_subgraph_nodes,
                         "edge_sample": edges[0] if edges else None,
                         "node_sample": nodes[0] if nodes else None},
            )
            raise

        stages[StageName.NEIGHBORS] = s3
        if not s3.is_ok():
            self._error_logger.log_stage_error(
                stage_name="neighbors", sample_id=inputs.sample_id,
                exception=Exception(f"Stage aborted: {s3.aborted}"),
                context={"query": inputs.query, "repo": inputs.repo,
                         "anchors": anchors, "aborted_reason": s3.aborted,
                         "stage_info": s3.info},
            )

        # При --until neighbors сохраняем sub_nodes/sub_edges — чтобы потом
        # можно было запустить только прунинг через --from-stage3.
        if s3.is_ok() and until == StageName.NEIGHBORS:
            _save_stage3(
                sample_id=inputs.sample_id,
                repo=inputs.repo,
                query=inputs.query,
                sub_nodes=s3.payload["sub_nodes"],
                sub_edges=s3.payload["sub_edges"],
                out_dir=self._settings.pipeline.outputs_dir / "stage3",
            )

        if not s3.is_ok() or until == StageName.NEIGHBORS:
            return _build_outcome(stages, inputs, node_by_id, edges)

        # ---- этап 4: LLM-прунинг --------------------------------------
        try:
            s4 = await stage4_prune.prune_subgraph(
                query=inputs.query,
                sub_nodes=s3.payload["sub_nodes"],
                sub_edges=s3.payload["sub_edges"],
                llm=self._llm,
                tracer=self._tracer,
                sample_id=inputs.sample_id,
            )
        except Exception as e:
            self._error_logger.log_stage_error(
                stage_name="prune", sample_id=inputs.sample_id, exception=e,
                context={"query": inputs.query, "repo": inputs.repo,
                         "sub_nodes_count": len(s3.payload["sub_nodes"]),
                         "sub_edges_count": len(s3.payload["sub_edges"])},
            )
            raise

        stages[StageName.PRUNE] = s4
        if not s4.is_ok():
            self._error_logger.log_stage_error(
                stage_name="prune", sample_id=inputs.sample_id,
                exception=Exception(f"Stage aborted: {s4.aborted}"),
                context={"query": inputs.query, "repo": inputs.repo,
                         "aborted_reason": s4.aborted, "stage_info": s4.info},
            )

        return _build_outcome(stages, inputs, node_by_id, edges)

    async def run(self, inputs: ApproachInputs) -> ApproachResult:
        """Минимальный интерфейс для реестра подходов."""
        outcome = await self.run_with_stages(inputs, until=StageName.PRUNE)
        return outcome.result

    async def aclose(self) -> None:
        return None


# ---- вспомогательные функции -------------------------------------------

def _save_stage3(
    *,
    sample_id: str,
    repo: str,
    query: str,
    sub_nodes: list[dict[str, Any]],
    sub_edges: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    """Сохранить подграф после этапа 3 для последующего запуска LLM."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "sample_id": sample_id,
        "repo": repo,
        "query": query,
        "sub_nodes": sub_nodes,
        "sub_edges": sub_edges,
    }
    with (out_dir / f"{sample_id}.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _build_outcome(
    stages: dict[StageName, StageOutcome],
    inputs: ApproachInputs,
    node_by_id: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> PipelineOutcome:
    """Собрать ApproachResult из последнего успешного этапа."""
    metadata: dict[str, Any] = {
        "stages_executed": [s.value for s in stages],
    }
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

    sub_nodes, sub_edges = filter_subgraph(list(node_by_id.values()), edges, keep)
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
    fname = diagram_filename_for_repo(repo or "")
    return fname[:-5] if fname.endswith(".json") else fname


def _is_jsonable(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool, list, dict)) or v is None
