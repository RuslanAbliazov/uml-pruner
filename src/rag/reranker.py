"""Local cross-encoder reranker.

Pairs a query with a list of candidate texts and returns a relevance score
for each pair. Unlike the bi-encoder in :mod:`src.rag.encoder`, a
cross-encoder reads the query and the candidate jointly, which usually gives
substantially better top-1 quality at the cost of being O(N) per query (no
precomputable index).

This module is a thin wrapper over ``sentence_transformers.CrossEncoder``:

* Lazy model loading (heavy deps imported only when first used) — same
  pattern as :class:`LocalEncoder`, so ``src.core``-level code stays free of
  torch/transformers imports.
* Automatic device selection via :func:`src.rag.encoder.detect_device`.
* Numpy float32 outputs.

The two main consumers are:

* approach #2 ``anchor_neighbors`` — picks the top-1 candidate as the anchor
  when ``anchor_selector: "reranker"`` is set in the YAML.
* future approaches that want a "rerank top-K" stage on top of an embedding
  retriever.

There is intentionally no caching layer here: rerank scores are query-
dependent (one pass per query), and the candidate list is already small
(top-K from the bi-encoder index), so we don't need a persistent cache like
:mod:`src.rag.cache`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.logger import get_logger
from src.rag.encoder import detect_device

logger = get_logger(__name__)

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


@dataclass
class RerankerConfig:
    """Knobs read from the top-level ``reranker:`` YAML section."""

    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "auto"  # 'auto' | 'cuda' | 'mps' | 'cpu'
    batch_size: int = 16
    # ``None`` means "use the model's own default". The CLI / YAML loader
    # translates ``-1`` and ``null`` into ``None``.
    max_seq_length: int | None = None
    trust_remote_code: bool = True


class LocalReranker:
    """Thin wrapper around ``sentence_transformers.CrossEncoder``.

    Usage::

        rr = LocalReranker(RerankerConfig(model_name="BAAI/bge-reranker-v2-m3"))
        scores = rr.score(query, [text_a, text_b, text_c])
        # scores is a 1-D float32 numpy array, higher = more relevant.

    The model is loaded on first call to :meth:`score`. Subsequent calls
    reuse it.
    """

    def __init__(self, cfg: RerankerConfig | None = None) -> None:
        self.cfg = cfg or RerankerConfig()
        self._model: "CrossEncoder | None" = None
        self._device: str | None = None

    # ---- lifecycle ------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. Run "
                "`pip install -r requirements-embeddings.txt`."
            ) from e

        self._device = detect_device(self.cfg.device)
        logger.info(
            "Loading reranker model '%s' on device '%s'",
            self.cfg.model_name,
            self._device,
        )
        kwargs: dict[str, Any] = {
            "device": self._device,
            "trust_remote_code": self.cfg.trust_remote_code,
        }
        if self.cfg.max_seq_length is not None:
            kwargs["max_length"] = self.cfg.max_seq_length
        self._model = CrossEncoder(self.cfg.model_name, **kwargs)
        logger.info("Reranker loaded.")

    @property
    def device(self) -> str:
        self._ensure_loaded()
        return self._device or "cpu"

    @property
    def model_name(self) -> str:
        return self.cfg.model_name

    # ---- scoring --------------------------------------------------------

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        """Score every (query, text) pair. Higher = more relevant.

        Returns a 1-D ``float32`` numpy array of length ``len(texts)``.
        Empty input returns an empty array without loading the model.
        """
        if not texts:
            return np.zeros((0,), dtype=np.float32)
        self._ensure_loaded()
        assert self._model is not None
        pairs = [(query, t) for t in texts]
        scores = self._model.predict(
            pairs,
            batch_size=self.cfg.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(scores, dtype=np.float32)

    # ---- introspection --------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Small dict describing the reranker — handy for run metadata."""
        self._ensure_loaded()
        return {
            "model_name": self.cfg.model_name,
            "device": self._device,
            "batch_size": self.cfg.batch_size,
            "max_seq_length": self.cfg.max_seq_length,
        }
