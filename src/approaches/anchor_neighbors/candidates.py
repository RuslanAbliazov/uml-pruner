"""Stage 1 — embedding-based candidate retrieval.

Loads (and caches) the on-disk embedding index, embeds the user query, and
returns the top-K most similar nodes from the diagram.
"""

from __future__ import annotations

from typing import Any, Optional

from src.core.logger import get_logger
from src.rag.cache import (
    EmbeddingCacheEntry,
    compute_diagram_hash,
    is_valid,
    load_cache,
)
from src.rag.encoder import EncoderConfig, LocalEncoder
from src.rag.retriever import retrieve_top_k

logger = get_logger(__name__)


def short_name(node_id: str) -> str:
    """Last dotted segment of a node_id (or the id itself if no dot)."""
    return node_id.rsplit(".", 1)[-1] if "." in node_id else node_id


class CandidateFinder:
    """Owns the embedding encoder + per-diagram index cache.

    Kept as a class (not a free function) so the runner can keep a
    long-lived encoder and avoid re-loading indices for every sample.
    """

    def __init__(
        self,
        *,
        model: str,
        device: str,
        batch_size: int,
        cache_dir: str,
    ) -> None:
        self._model = model
        self._device = device
        self._batch_size = batch_size
        self._cache_dir = cache_dir
        self._encoder: Optional[LocalEncoder] = None
        self._index_by_stem: dict[str, EmbeddingCacheEntry] = {}

    def fetch(
        self,
        *,
        query: str,
        diagram_stem: str,
        nodes: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Return up to ``top_k`` candidate dicts.

        Each dict carries the keys the LLM needs to choose an anchor:
        ``node_id``, ``name``, ``type``, ``score``.
        """
        entry = self._load_index(diagram_stem, nodes)
        if entry is None:
            logger.warning(
                "anchor_neighbors: no usable embedding index for '%s' "
                "(run scripts/build_index.py first)",
                diagram_stem,
            )
            return []

        encoder = self._get_encoder()
        hits = retrieve_top_k(query, entry, encoder, top_k=top_k)
        node_by_id = {n["node_id"]: n for n in nodes if n.get("node_id")}

        out: list[dict[str, Any]] = []
        for h in hits:
            node = node_by_id.get(h.node_id)
            if node is None:
                # Index may be stale — orphan ids are skipped defensively.
                continue
            out.append(
                {
                    "node_id": h.node_id,
                    "name": node.get("name") or short_name(h.node_id),
                    "type": node.get("type", "class"),
                    "score": round(float(h.score), 6),
                }
            )
        return out

    # ------------------------------------------------------------------

    def _get_encoder(self) -> LocalEncoder:
        if self._encoder is None:
            self._encoder = LocalEncoder(
                EncoderConfig(
                    model_name=self._model,
                    device=self._device,
                    batch_size=self._batch_size,
                )
            )
        return self._encoder

    def _load_index(
        self, diagram_stem: str, nodes: list[dict[str, Any]]
    ) -> Optional[EmbeddingCacheEntry]:
        cached = self._index_by_stem.get(diagram_stem)
        if cached is not None:
            return cached

        entry = load_cache(self._cache_dir, diagram_stem)
        if entry is None:
            return None
        if not is_valid(
            entry,
            expected_model=self._model,
            expected_diagram_hash=compute_diagram_hash(nodes),
        ):
            logger.warning(
                "anchor_neighbors: cache for '%s' is stale; rebuild with "
                "scripts/build_index.py --force",
                diagram_stem,
            )
            return None
        self._index_by_stem[diagram_stem] = entry
        return entry
