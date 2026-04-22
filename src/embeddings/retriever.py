"""Top-K retrieval by cosine similarity.

Assumes both the cache vectors and the query vector are L2-normalized
(the default when using LocalEncoder). In that case, cosine similarity equals
the dot product, so ranking via `vectors @ query` is exact and fast.

For K <= 1000 and N <= 50_000, numpy is more than enough — no need for FAISS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.embeddings.cache import EmbeddingCacheEntry
from src.embeddings.encoder import LocalEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalHit:
    node_id: str
    score: float


def top_k_by_vector(
    query_vec: np.ndarray,
    entry: EmbeddingCacheEntry,
    top_k: int,
) -> list[RetrievalHit]:
    """Return the top-k node_ids ranked by dot product against `query_vec`.

    If vectors are L2-normalized, dot product == cosine similarity.
    """
    if entry.vectors.shape[0] == 0:
        return []
    k = min(top_k, entry.vectors.shape[0])

    # Dot product against all vectors: shape (N,)
    scores = entry.vectors @ query_vec.astype(np.float32, copy=False)

    # argpartition for cheap top-k selection, then argsort inside the slice
    if k < len(scores):
        part = np.argpartition(-scores, k - 1)[:k]
    else:
        part = np.arange(len(scores))
    # sort the selected indices by descending score
    order = part[np.argsort(-scores[part])]

    return [
        RetrievalHit(node_id=entry.node_ids[i], score=float(scores[i])) for i in order
    ]


def retrieve_top_k(
    query: str,
    entry: EmbeddingCacheEntry,
    encoder: LocalEncoder,
    top_k: int = 300,
) -> list[RetrievalHit]:
    """Encode the query and return top-k hits."""
    query_vec = encoder.encode_query(query)
    hits = top_k_by_vector(query_vec, entry, top_k)
    logger.info(
        "Retrieved top-%d nodes (score range: %.3f .. %.3f)",
        len(hits),
        hits[-1].score if hits else 0.0,
        hits[0].score if hits else 0.0,
    )
    return hits


def retrieve_top_k_ids(
    query: str,
    entry: EmbeddingCacheEntry,
    encoder: LocalEncoder,
    top_k: int = 300,
) -> set[str]:
    """Convenience wrapper: return just the set of node_ids."""
    return {h.node_id for h in retrieve_top_k(query, entry, encoder, top_k)}
