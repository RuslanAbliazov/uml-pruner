"""Approach #2: anchor selection + neighborhood expansion + LLM prune.

Pipeline shape:

    1. **Candidates.** Use the embedding retriever to fetch the top-K classes
       most similar to the query (default K=10, configurable). The diagram
       must already have a pre-built embedding index on disk
       (``data/embeddings/<diagram_stem>``).
    2. **Anchor.** Ask the LLM to pick the SINGLE best anchor among those
       candidates. The anchor is the class from which we will explore the
       graph.
    3. **Expand.** Collect every neighbor of the anchor — both outgoing and
       incoming edges, every relation kind. Self-loops are discarded.
    4. **Prune.** Pass the anchor-centered subgraph (anchor + all neighbors,
       plus the edges that touch the anchor) to the LLM and ask it to
       classify each node as REQUIRED / USEFUL / IRRELEVANT.

The result is a pruned ``{nodes, edges}`` subgraph (REQUIRED ∪ USEFUL plus
edges between them) wrapped in :class:`ApproachResult`.

Persistence is done by the caller (``scripts/benchmark.py``); the runner
itself never touches the filesystem.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.approaches.base import ApproachInputs, ApproachResult
from src.embeddings.cache import (
    EmbeddingCacheEntry,
    compute_diagram_hash,
    is_valid,
    load_cache,
)
from src.embeddings.encoder import EncoderConfig, LocalEncoder
from src.embeddings.retriever import retrieve_top_k
from src.llm.client import LLMClient
from src.llm.parser import parse_json_response
from src.llm.prompts import (
    anchor_prune_system_prompt,
    anchor_select_system_prompt,
    build_anchor_prune_user_prompt,
    build_anchor_select_user_prompt,
)
from src.preprocessing.compressor import filter_subgraph
from src.utils.logger import get_logger

NAME = "anchor_neighbors"

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AnchorNeighborsConfig:
    """Knobs for approach #2.

    Sourced from ``configs/config.yaml`` (``approaches.anchor_neighbors``)
    via :func:`build_runner`. Defaults match the design described in the
    approach README.
    """

    # ---- candidate generation (RAG) --------------------------------------
    n_candidates: int = 10  # how many candidates the retriever returns
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "auto"
    embedding_batch_size: int = 8
    embedding_cache_dir: str = "data/embeddings"

    # ---- prune stage -----------------------------------------------------
    # Hard cap on neighborhood size shipped to the LLM. Defends against
    # extremely high-degree anchors blowing the context window. Set to 0
    # to disable.
    max_subgraph_nodes: int = 200


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class AnchorNeighborsRunner:
    """Implements approach #2 against the unified :class:`ApproachRunner` interface."""

    name = NAME

    def __init__(self, cfg: AnchorNeighborsConfig, llm_client: LLMClient) -> None:
        self._cfg = cfg
        self._llm = llm_client
        # Lazily-initialised resources.
        self._encoder: Optional[LocalEncoder] = None
        # Per-diagram cached embedding index. Key: diagram stem.
        self._index_cache: dict[str, EmbeddingCacheEntry] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, inputs: ApproachInputs) -> ApproachResult:
        nodes = inputs.nodes
        edges = inputs.edges
        node_by_id = {n["node_id"]: n for n in nodes if n.get("node_id")}
        diagram_stem = self._diagram_stem(inputs)

        # 1. Embedding retrieval ---------------------------------------------------
        candidates = self._fetch_candidates(
            inputs.query, diagram_stem, nodes, top_k=self._cfg.n_candidates
        )
        if not candidates:
            return self._empty_result(
                reason="no_candidates",
                metadata={"diagram_stem": diagram_stem},
            )

        # 2. LLM picks the anchor --------------------------------------------------
        anchor_id, anchor_reason = await self._select_anchor(
            inputs.query, candidates, node_by_id
        )
        if not anchor_id:
            return self._empty_result(
                reason="anchor_selection_failed",
                metadata={
                    "diagram_stem": diagram_stem,
                    "candidates": [c["node_id"] for c in candidates],
                },
            )

        # 3. Expand neighbors ------------------------------------------------------
        neighbor_ids = self._neighbors_of(anchor_id, edges)
        subgraph_ids: set[str] = {anchor_id} | neighbor_ids
        # Defensive cap on the subgraph size before we ship it to the LLM.
        truncated = False
        if (
            self._cfg.max_subgraph_nodes
            and len(subgraph_ids) > self._cfg.max_subgraph_nodes
        ):
            truncated = True
            subgraph_ids = self._trim_subgraph(
                anchor_id, neighbor_ids, edges, self._cfg.max_subgraph_nodes
            )

        sub_nodes = [node_by_id[nid] for nid in subgraph_ids if nid in node_by_id]
        sub_edges = [
            e
            for e in edges
            if e.get("node_id_from") in subgraph_ids
            and e.get("node_id_to") in subgraph_ids
        ]

        # 4. LLM prune --------------------------------------------------------------
        required_ids, useful_ids = await self._prune(
            inputs.query, anchor_id, sub_nodes, sub_edges
        )
        # The anchor itself is implicitly relevant — guarantee it's kept somewhere.
        if anchor_id not in required_ids and anchor_id not in useful_ids:
            required_ids.add(anchor_id)

        kept_ids = required_ids | useful_ids
        final_nodes, final_edges = filter_subgraph(nodes, edges, kept_ids)

        return ApproachResult(
            approach=self.name,
            nodes=final_nodes,
            edges=final_edges,
            required_node_ids=sorted(required_ids),
            useful_node_ids=sorted(useful_ids),
            metadata={
                "diagram_stem": diagram_stem,
                "n_candidates_requested": self._cfg.n_candidates,
                "n_candidates_returned": len(candidates),
                "candidates": [
                    {"node_id": c["node_id"], "score": c.get("score")}
                    for c in candidates
                ],
                "anchor": anchor_id,
                "anchor_reason": anchor_reason,
                "neighbors_count": len(neighbor_ids),
                "subgraph_node_count": len(sub_nodes),
                "subgraph_edge_count": len(sub_edges),
                "subgraph_truncated": truncated,
                "filtered_node_count": len(final_nodes),
            },
        )

    async def aclose(self) -> None:
        # LLMClient has no long-lived connection; encoder uses local files only.
        return None

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _fetch_candidates(
        self,
        query: str,
        diagram_stem: str,
        nodes: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Run embedding retrieval; return up to ``top_k`` candidates as dicts."""
        entry = self._load_index(diagram_stem, nodes)
        if entry is None:
            logger.warning(
                "anchor_neighbors: no usable embedding index for '%s' "
                "(run scripts/build_index.py first)",
                diagram_stem,
            )
            return []

        if self._encoder is None:
            self._encoder = LocalEncoder(
                EncoderConfig(
                    model_name=self._cfg.embedding_model,
                    device=self._cfg.embedding_device,
                    batch_size=self._cfg.embedding_batch_size,
                )
            )

        hits = retrieve_top_k(query, entry, self._encoder, top_k=top_k)
        node_by_id = {n["node_id"]: n for n in nodes if n.get("node_id")}

        out: list[dict[str, Any]] = []
        for h in hits:
            node = node_by_id.get(h.node_id)
            if node is None:
                # The index may be stale; skip orphan ids defensively.
                continue
            out.append(
                {
                    "node_id": h.node_id,
                    "name": node.get("name") or _short_name(h.node_id),
                    "type": node.get("type", "class"),
                    "score": round(float(h.score), 6),
                }
            )
        return out

    async def _select_anchor(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        node_by_id: dict[str, dict[str, Any]],
    ) -> tuple[Optional[str], str]:
        """Ask the LLM to pick one anchor from ``candidates``.

        Returns ``(anchor_id, reason)``. ``anchor_id`` is None if the LLM
        misbehaves (returns nothing valid); we then short-circuit upstream.
        """
        # Enrich candidates with a compact method preview to help the LLM.
        enriched: list[dict[str, Any]] = []
        for c in candidates:
            node = node_by_id.get(c["node_id"]) or {}
            method_preview = _method_preview(node.get("methods") or [])
            enriched.append(
                {
                    "node_id": c["node_id"],
                    "name": c.get("name"),
                    "type": c.get("type"),
                    "score": c.get("score"),
                    "methods_preview": method_preview,
                }
            )

        user_prompt = build_anchor_select_user_prompt(query=query, candidates=enriched)
        t0 = time.time()
        try:
            resp = await self._llm.call(
                anchor_select_system_prompt(),
                user_prompt,
                json_mode=True,
            )
        except Exception:
            logger.exception("anchor selection LLM call failed")
            return None, ""
        logger.debug(
            "anchor selection: %.2fs, %d in / %d out tokens",
            time.time() - t0,
            resp.input_tokens,
            resp.output_tokens,
        )

        try:
            data = parse_json_response(resp.content)
        except ValueError:
            logger.warning("anchor selection: could not parse LLM JSON")
            return None, ""
        if not isinstance(data, dict):
            return None, ""

        anchor = data.get("anchor")
        reason = (data.get("reason") or "").strip()
        valid_ids = {c["node_id"] for c in candidates}
        if isinstance(anchor, str) and anchor in valid_ids:
            return anchor, reason
        # Hallucinated anchor → fall back to the top-1 candidate as a safety net.
        logger.warning(
            "anchor selection: LLM returned invalid/missing anchor (%r); "
            "falling back to top-1 candidate",
            anchor,
        )
        return candidates[0]["node_id"], "fallback: LLM returned invalid anchor"

    @staticmethod
    def _neighbors_of(anchor: str, edges: list[dict[str, Any]]) -> set[str]:
        """Every neighbor of ``anchor``, regardless of edge direction or kind."""
        out: set[str] = set()
        for e in edges:
            src = e.get("node_id_from")
            dst = e.get("node_id_to")
            if src == anchor and dst and dst != anchor:
                out.add(dst)
            elif dst == anchor and src and src != anchor:
                out.add(src)
        return out

    @staticmethod
    def _trim_subgraph(
        anchor: str,
        neighbors: set[str],
        edges: list[dict[str, Any]],
        cap: int,
    ) -> set[str]:
        """When a degree explosion happens, keep the ``cap-1`` neighbors with
        the highest edge multiplicity to the anchor (anchor itself always kept).

        Higher edge count = stronger structural relation, so this favors the
        most-connected neighbors.
        """
        scores: dict[str, int] = {}
        for e in edges:
            src = e.get("node_id_from")
            dst = e.get("node_id_to")
            if src == anchor and dst in neighbors:
                scores[dst] = scores.get(dst, 0) + 1
            elif dst == anchor and src in neighbors:
                scores[src] = scores.get(src, 0) + 1
        ranked = sorted(neighbors, key=lambda n: (-scores.get(n, 0), n))
        return {anchor, *ranked[: max(0, cap - 1)]}

    async def _prune(
        self,
        query: str,
        anchor: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> tuple[set[str], set[str]]:
        """Ask the LLM to classify the anchor-centered subgraph."""
        # Trim the node payload to the LLM-friendly subset (no description leak).
        trimmed_nodes = [_node_for_llm(n) for n in nodes]
        trimmed_edges = [_edge_for_llm(e) for e in edges]
        user_prompt = build_anchor_prune_user_prompt(
            query=query,
            anchor=anchor,
            nodes=trimmed_nodes,
            edges=trimmed_edges,
        )
        t0 = time.time()
        try:
            resp = await self._llm.call(
                anchor_prune_system_prompt(),
                user_prompt,
                json_mode=True,
            )
        except Exception:
            logger.exception("anchor prune LLM call failed")
            return set(), set()
        logger.debug(
            "anchor prune: %.2fs, %d in / %d out tokens",
            time.time() - t0,
            resp.input_tokens,
            resp.output_tokens,
        )

        try:
            data = parse_json_response(resp.content)
        except ValueError:
            logger.warning("anchor prune: could not parse LLM JSON")
            return set(), set()
        if not isinstance(data, dict):
            return set(), set()

        valid_ids = {n["node_id"] for n in nodes if n.get("node_id")}
        required = {
            x for x in (data.get("required") or []) if isinstance(x, str) and x in valid_ids
        }
        useful = {
            x
            for x in (data.get("useful") or [])
            if isinstance(x, str) and x in valid_ids and x not in required
        }
        return required, useful

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_index(
        self, diagram_stem: str, nodes: list[dict[str, Any]]
    ) -> Optional[EmbeddingCacheEntry]:
        """Load (and validate) the on-disk embedding index for ``diagram_stem``.

        Caches loaded indices in memory across samples for the same diagram.
        Returns None if the cache is missing OR the model/diagram-hash does
        not match the runtime config.
        """
        if diagram_stem in self._index_cache:
            return self._index_cache[diagram_stem]

        entry = load_cache(self._cfg.embedding_cache_dir, diagram_stem)
        if entry is None:
            return None
        expected_hash = compute_diagram_hash(nodes)
        if not is_valid(
            entry,
            expected_model=self._cfg.embedding_model,
            expected_diagram_hash=expected_hash,
        ):
            logger.warning(
                "anchor_neighbors: cache for '%s' is stale "
                "(model or diagram changed); rebuild with build_index.py --force",
                diagram_stem,
            )
            return None
        self._index_cache[diagram_stem] = entry
        return entry

    @staticmethod
    def _diagram_stem(inputs: ApproachInputs) -> str:
        """Resolve the diagram filename stem used to locate the embedding index.

        Convention: the benchmark loads diagrams from
        ``<diagrams_dir>/<repo_basename>.json``. The matching index lives
        under ``<embeddings_cache>/<repo_basename>/``.
        """
        if inputs.repo and "/" in inputs.repo:
            return inputs.repo.split("/", 1)[1]
        return inputs.repo or ""

    def _empty_result(
        self, *, reason: str, metadata: dict[str, Any]
    ) -> ApproachResult:
        """Build an empty result with diagnostic metadata."""
        meta = dict(metadata)
        meta["aborted"] = reason
        return ApproachResult(
            approach=self.name,
            nodes=[],
            edges=[],
            required_node_ids=[],
            useful_node_ids=[],
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# Helpers (module-level)
# ---------------------------------------------------------------------------


def _short_name(node_id: str) -> str:
    return node_id.rsplit(".", 1)[-1] if "." in node_id else node_id


def _method_preview(methods: list[str], limit: int = 8) -> list[str]:
    """Return a short list of method names (no params) to give the LLM a hint
    of what each candidate class does.
    """
    seen: set[str] = set()
    out: list[str] = []
    for m in methods:
        head = m.split("(", 1)[0].strip()
        if not head or head in seen:
            continue
        seen.add(head)
        out.append(head)
        if len(out) >= limit:
            break
    return out


def _node_for_llm(node: dict[str, Any]) -> dict[str, Any]:
    """Project a node to the minimal info we want the prune LLM to see.

    We deliberately drop ``description`` (LLM-generated, would leak into the
    answer) and trim methods/params to keep the prompt compact.
    """
    return {
        "node_id": node.get("node_id"),
        "name": node.get("name") or _short_name(node.get("node_id", "")),
        "type": node.get("type", "class"),
        "methods": (node.get("methods") or [])[:30],
        "params": (node.get("params") or [])[:20],
    }


def _edge_for_llm(edge: dict[str, Any]) -> dict[str, Any]:
    """Edge stripped to the fields useful for pruning judgments."""
    return {
        "from": edge.get("node_id_from"),
        "to": edge.get("node_id_to"),
        "kind": edge.get("description") or edge.get("kind") or "",
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_runner(cfg: Any | None = None) -> AnchorNeighborsRunner:
    """Construct the runner from the project YAML config.

    Reads:
        - ``approaches.anchor_neighbors`` for approach-specific knobs (with
          sensible defaults if the section is absent).
        - ``embeddings`` for shared retrieval defaults (model, device, etc.).
        - ``llm`` for the LLM connection.
    """
    from src.utils.config import load_config  # avoid import cycles at module load

    if cfg is None:
        cfg = load_config("configs/config.yaml")

    llm_cfg = cfg.llm
    emb_raw = cfg.get("embeddings") if hasattr(cfg, "get") else None
    approach_raw = (
        cfg.get("approaches") if hasattr(cfg, "get") and cfg.get("approaches") else None
    )
    section_raw = (
        approach_raw.get("anchor_neighbors")
        if approach_raw and hasattr(approach_raw, "get")
        else None
    )

    def _emb(key: str, default: Any) -> Any:
        if emb_raw is None:
            return default
        try:
            return emb_raw.get(key, default)
        except AttributeError:
            return default

    def _approach(key: str, default: Any) -> Any:
        if section_raw is None:
            return default
        try:
            return section_raw.get(key, default)
        except AttributeError:
            return default

    cache_dir = _emb("cache_dir", "data/embeddings") or "data/embeddings"

    def _coerce_cap(value: Any, default: int) -> int:
        """Normalize an integer cap. ``None`` / negative -> 0 (= disabled)."""
        if value is None:
            return 0
        try:
            n = int(value)
        except (TypeError, ValueError):
            return default
        return n if n > 0 else 0

    runner_cfg = AnchorNeighborsConfig(
        n_candidates=int(_approach("n_candidates", 10) or 10),
        embedding_model=_emb("model", "BAAI/bge-m3") or "BAAI/bge-m3",
        embedding_device=_emb("device", "auto") or "auto",
        embedding_batch_size=int(_emb("batch_size", 8) or 8),
        embedding_cache_dir=cache_dir if isinstance(cache_dir, str) else "data/embeddings",
        max_subgraph_nodes=_coerce_cap(_approach("max_subgraph_nodes", 200), 200),
    )

    client = LLMClient(
        model=llm_cfg.get("model", "gpt-4-turbo-preview"),
        temperature=llm_cfg.get("temperature", 0.1),
        max_tokens=llm_cfg.get("max_tokens", 4096),
        timeout=llm_cfg.get("timeout", 90),
        retry_attempts=llm_cfg.get("retry_attempts", 3),
        retry_delay=llm_cfg.get("retry_delay", 2),
        api_key=llm_cfg.get("api_key", "") or None,
        base_url=llm_cfg.get("base_url", "") or None,
    )

    return AnchorNeighborsRunner(runner_cfg, client)
