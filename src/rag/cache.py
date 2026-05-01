"""On-disk cache for computed node embeddings.

Layout (per diagram):

    data/embeddings/<diagram_name>/
        meta.json        # model info, diagram hash, node count, dimension
        node_ids.json    # [node_id_1, node_id_2, ...] in the same order as vectors
        vectors.npy      # float32 array of shape (N, D), L2-normalized if configured

Invalidation:
- If the diagram's node set or the model changes, the cache is considered stale
  and needs to be rebuilt.
- `is_valid()` checks both.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.core.logger import get_logger

logger = get_logger(__name__)


META_FILE = "meta.json"
NODE_IDS_FILE = "node_ids.json"
VECTORS_FILE = "vectors.npy"


@dataclass
class EmbeddingCacheEntry:
    """In-memory representation of a loaded cache."""

    model_name: str
    dimension: int
    node_ids: list[str]
    vectors: np.ndarray  # shape (N, D), float32
    diagram_hash: str

    def __len__(self) -> int:
        return len(self.node_ids)


# -----------------------------------------------------------------------------
# Diagram hashing
# -----------------------------------------------------------------------------


def compute_diagram_hash(nodes: list[dict[str, Any]]) -> str:
    """Compute a stable hash over the node set.

    We only hash node_ids + a fingerprint of each node's content (methods +
    description) so that minor metadata changes invalidate the cache, but
    reorderings of the list do not (we sort first).
    """
    fingerprints = []
    for n in nodes:
        nid = n.get("node_id", "")
        methods = n.get("methods") or []
        desc = (n.get("description") or "")[:200]
        fingerprints.append(f"{nid}\x00{len(methods)}\x00{desc}")
    fingerprints.sort()
    h = hashlib.sha256()
    for f in fingerprints:
        h.update(f.encode("utf-8"))
        h.update(b"\n")
    return "sha256:" + h.hexdigest()


# -----------------------------------------------------------------------------
# Cache I/O
# -----------------------------------------------------------------------------


def cache_dir_for(base_dir: str | Path, diagram_name: str) -> Path:
    """Return the cache directory for a given diagram (does not create it)."""
    return Path(base_dir) / diagram_name


def save_cache(
    base_dir: str | Path,
    diagram_name: str,
    model_name: str,
    node_ids: list[str],
    vectors: np.ndarray,
    diagram_hash: str,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Persist an embedding cache to disk. Overwrites any existing cache."""
    target = cache_dir_for(base_dir, diagram_name)
    target.mkdir(parents=True, exist_ok=True)

    if vectors.shape[0] != len(node_ids):
        raise ValueError(
            f"vectors count ({vectors.shape[0]}) != node_ids count ({len(node_ids)})"
        )

    meta: dict[str, Any] = {
        "diagram_name": diagram_name,
        "model_name": model_name,
        "diagram_hash": diagram_hash,
        "node_count": int(vectors.shape[0]),
        "dimension": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "dtype": str(vectors.dtype),
    }
    if extra_meta:
        meta.update(extra_meta)

    (target / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (target / NODE_IDS_FILE).write_text(
        json.dumps(node_ids, ensure_ascii=False), encoding="utf-8"
    )
    np.save(target / VECTORS_FILE, vectors)
    logger.info(
        "Saved embedding cache: %s (%d nodes x %d dim)",
        target,
        meta["node_count"],
        meta["dimension"],
    )
    return target


def load_cache(
    base_dir: str | Path,
    diagram_name: str,
) -> EmbeddingCacheEntry | None:
    """Load an embedding cache. Returns None if not present or corrupt."""
    target = cache_dir_for(base_dir, diagram_name)
    meta_path = target / META_FILE
    ids_path = target / NODE_IDS_FILE
    vecs_path = target / VECTORS_FILE

    if not (meta_path.exists() and ids_path.exists() and vecs_path.exists()):
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        node_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        vectors = np.load(vecs_path)
    except (json.JSONDecodeError, ValueError, OSError) as e:
        logger.warning("Could not load embedding cache at %s: %s", target, e)
        return None

    if vectors.shape[0] != len(node_ids):
        logger.warning(
            "Cache at %s is inconsistent: %d vectors vs %d node_ids",
            target,
            vectors.shape[0],
            len(node_ids),
        )
        return None

    return EmbeddingCacheEntry(
        model_name=meta.get("model_name", ""),
        dimension=int(
            meta.get("dimension", vectors.shape[1] if vectors.ndim == 2 else 0)
        ),
        node_ids=node_ids,
        vectors=vectors.astype(np.float32, copy=False),
        diagram_hash=meta.get("diagram_hash", ""),
    )


def is_valid(
    entry: EmbeddingCacheEntry,
    expected_model: str,
    expected_diagram_hash: str,
) -> bool:
    """Check whether a loaded cache matches the current model and diagram."""
    if entry.model_name != expected_model:
        logger.info(
            "Cache model mismatch: cached=%s, expected=%s",
            entry.model_name,
            expected_model,
        )
        return False
    if entry.diagram_hash != expected_diagram_hash:
        logger.info("Cache diagram hash mismatch (diagram changed since indexing)")
        return False
    return True
