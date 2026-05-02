"""Local embedding encoder using sentence-transformers.

Supports swapping models via config. Default is nomic-embed-text-v1.5, but any
model that sentence-transformers can load will work. Some models (like nomic)
require task-specific prefixes on inputs; we handle those cases automatically.

Device selection is automatic: CUDA > MPS (Apple Silicon) > CPU.
The heavy dependencies (torch, sentence-transformers) are imported lazily so
that the core pipeline does not require them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.logger import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


# -----------------------------------------------------------------------------
# Model-specific prefix handling
# -----------------------------------------------------------------------------
# Some embedding models require prefixes on queries vs documents to get good
# retrieval quality. We keep a small registry and fall back to "no prefix".

_PREFIX_REGISTRY: dict[str, tuple[str, str]] = {
    # (document_prefix, query_prefix)
    "nomic-ai/nomic-embed-text-v1.5": ("search_document: ", "search_query: "),
    "nomic-ai/nomic-embed-text-v1": ("search_document: ", "search_query: "),
    "BAAI/bge-small-en-v1.5": (
        "",
        "Represent this sentence for searching relevant passages: ",
    ),
    "BAAI/bge-base-en-v1.5": (
        "",
        "Represent this sentence for searching relevant passages: ",
    ),
    "BAAI/bge-large-en-v1.5": (
        "",
        "Represent this sentence for searching relevant passages: ",
    ),
    "intfloat/e5-small-v2": ("passage: ", "query: "),
    "intfloat/e5-base-v2": ("passage: ", "query: "),
    "intfloat/e5-large-v2": ("passage: ", "query: "),
    "intfloat/multilingual-e5-large": ("passage: ", "query: "),
}


def _prefixes_for(model_name: str) -> tuple[str, str]:
    """Return (document_prefix, query_prefix) for a model; empty strings if unknown."""
    return _PREFIX_REGISTRY.get(model_name, ("", ""))


# -----------------------------------------------------------------------------
# Device detection
# -----------------------------------------------------------------------------


def detect_device(preference: str = "auto") -> str:
    """Pick a torch device.

    Args:
        preference: 'auto' | 'cuda' | 'mps' | 'cpu'.

    Returns:
        Device string usable by torch / sentence-transformers.
    """
    preference = (preference or "auto").lower()
    if preference != "auto":
        return preference

    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for local embeddings. Install it via "
            "`pip install -r requirements-embeddings.txt`."
        ) from e

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# -----------------------------------------------------------------------------
# Encoder
# -----------------------------------------------------------------------------


@dataclass
class EncoderConfig:
    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    device: str = "auto"  # 'auto' | 'cuda' | 'mps' | 'cpu'
    batch_size: int = 32
    max_seq_length: int | None = None  # None -> model default
    normalize: bool = True  # L2-normalize vectors (for cosine sim)
    trust_remote_code: bool = True  # nomic-embed requires this


class LocalEncoder:
    """Wrapper over sentence-transformers with:

    - lazy model loading
    - automatic device selection
    - model-specific prefix handling for documents vs queries
    - numpy output (float32, optionally L2-normalized)
    """

    def __init__(self, cfg: EncoderConfig | None = None):
        self.cfg = cfg or EncoderConfig()
        self._model: "SentenceTransformer | None" = None
        self._device: str | None = None
        self._doc_prefix, self._query_prefix = _prefixes_for(self.cfg.model_name)

    # --- lifecycle -------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. Run "
                "`pip install -r requirements-embeddings.txt`."
            ) from e

        self._device = detect_device(self.cfg.device)
        logger.info(
            "Loading embedding model '%s' on device '%s'",
            self.cfg.model_name,
            self._device,
        )
        self._model = SentenceTransformer(
            self.cfg.model_name,
            device=self._device,
            trust_remote_code=self.cfg.trust_remote_code,
        )
        if self.cfg.max_seq_length is not None:
            self._model.max_seq_length = self.cfg.max_seq_length
        logger.info(
            "Model loaded. max_seq_length=%s, embedding_dim=%s",
            self._model.max_seq_length,
            self._model.get_embedding_dimension(),
        )

    @property
    def device(self) -> str:
        self._ensure_loaded()
        return self._device or "cpu"

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        assert self._model is not None
        return int(self._model.get_embedding_dimension())

    @property
    def model_name(self) -> str:
        return self.cfg.model_name

    # --- encoding --------------------------------------------------------

    def encode_documents(
        self,
        texts: list[str],
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """Encode document texts (adds document prefix if the model requires it)."""
        self._ensure_loaded()
        assert self._model is not None
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        prefixed = [self._doc_prefix + t for t in texts] if self._doc_prefix else texts
        vectors = self._model.encode(
            prefixed,
            batch_size=self.cfg.batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.cfg.normalize,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32, copy=False)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string (adds query prefix if needed).

        Returns a 1-D float32 array.
        """
        self._ensure_loaded()
        assert self._model is not None
        text = (self._query_prefix + query) if self._query_prefix else query
        vec = self._model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=self.cfg.normalize,
            convert_to_numpy=True,
        )[0]
        return vec.astype(np.float32, copy=False)

    # --- introspection ---------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Return a dict describing the encoder (for cache metadata)."""
        self._ensure_loaded()
        return {
            "model_name": self.cfg.model_name,
            "device": self._device,
            "dimension": self.dimension,
            "max_seq_length": self._model.max_seq_length if self._model else None,
            "normalize": self.cfg.normalize,
            "doc_prefix": self._doc_prefix,
            "query_prefix": self._query_prefix,
        }
