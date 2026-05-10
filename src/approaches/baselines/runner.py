"""Baseline runners.

Two flavours:

* Query-agnostic (``empty``, ``full_diagram``, ``random_subset``, ``top_degree``)
  — no model, no LLM, no query inspection. Floor/ceiling references.
* Lexical (``bm25``) — uses BM25 over the same node-text serialization as
  the embedding retriever, so dense retrieval can be compared directly to
  classical sparse retrieval. Pulls in ``rank_bm25`` (small pure-Python).

Why include these in a serious benchmark: they make claims like "our
approach reaches F1 = 0.40" interpretable. If ``random_subset`` of the same
size already reaches F1 = 0.35, the "real" approach contributes very little.
If ``bm25`` matches your dense retriever's score, the embedding work isn't
pulling its weight on this dataset.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from typing import Any

from src.core.types import ApproachInputs, ApproachResult


# ---- helpers ---------------------------------------------------------------


def _all_node_ids(inputs: ApproachInputs) -> list[str]:
    """Return all valid node_ids in a deterministic order (insertion-order)."""
    out: list[str] = []
    seen: set[str] = set()
    for n in inputs.nodes:
        nid = n.get("node_id")
        if isinstance(nid, str) and nid and nid not in seen:
            out.append(nid)
            seen.add(nid)
    return out


def _select_subgraph(
    inputs: ApproachInputs, keep: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter nodes/edges down to ``keep``. Edges retained iff both ends kept."""
    nodes_out = [n for n in inputs.nodes if n.get("node_id") in keep]
    edges_out = [
        e
        for e in inputs.edges
        if e.get("node_id_from") in keep and e.get("node_id_to") in keep
    ]
    return nodes_out, edges_out


def _stable_seed(base_seed: int, sample_id: str) -> int:
    """Combine a base seed with a sample_id into a stable per-sample seed.

    We hash the sample_id so that rerunning the same sample is bit-exact
    even across machines, and so that the result doesn't depend on Python's
    hash randomization (PYTHONHASHSEED).
    """
    h = hashlib.blake2b(sample_id.encode("utf-8"), digest_size=8).hexdigest()
    return base_seed ^ int(h, 16)


# ---- runners ---------------------------------------------------------------


class EmptyBaseline:
    """Predicts the empty subgraph. Recall = 0, precision undefined (= 0)."""

    name = "empty"

    async def run(self, inputs: ApproachInputs) -> ApproachResult:
        return ApproachResult(
            approach=self.name,
            nodes=[],
            edges=[],
            required_node_ids=[],
            useful_node_ids=[],
            metadata={"strategy": "empty"},
        )

    async def aclose(self) -> None:
        return None


class FullDiagramBaseline:
    """Predicts the entire diagram. Recall = 1.0; precision tiny.

    The output ``required_node_ids`` is the full node set; ``useful_node_ids``
    is empty (the split is meaningless here, but we have to pick one).
    """

    name = "full_diagram"

    async def run(self, inputs: ApproachInputs) -> ApproachResult:
        all_ids = _all_node_ids(inputs)
        return ApproachResult(
            approach=self.name,
            nodes=list(inputs.nodes),
            edges=list(inputs.edges),
            required_node_ids=sorted(all_ids),
            useful_node_ids=[],
            metadata={
                "strategy": "full_diagram",
                "n_nodes": len(all_ids),
                "n_edges": len(inputs.edges),
            },
        )

    async def aclose(self) -> None:
        return None


class RandomSubsetBaseline:
    """Predicts ``size`` nodes chosen uniformly at random per sample.

    Seeded by ``(base_seed, sample_id)`` for reproducibility. Two runs with
    the same dataset produce the same predictions.
    """

    name = "random_subset"

    def __init__(self, size: int = 5, seed: int = 42) -> None:
        if size < 0:
            raise ValueError(f"size must be >= 0, got {size}")
        self._size = size
        self._seed = seed

    async def run(self, inputs: ApproachInputs) -> ApproachResult:
        ids = _all_node_ids(inputs)
        if not ids or self._size == 0:
            return ApproachResult(
                approach=self.name,
                metadata={
                    "strategy": "random_subset",
                    "size": self._size,
                    "n_available": len(ids),
                },
            )
        k = min(self._size, len(ids))
        rng = random.Random(_stable_seed(self._seed, inputs.sample_id))
        chosen = set(rng.sample(ids, k=k))
        nodes_out, edges_out = _select_subgraph(inputs, chosen)
        return ApproachResult(
            approach=self.name,
            nodes=nodes_out,
            edges=edges_out,
            required_node_ids=sorted(chosen),
            useful_node_ids=[],
            metadata={
                "strategy": "random_subset",
                "size": self._size,
                "n_available": len(ids),
                "n_picked": k,
                "seed_base": self._seed,
            },
        )

    async def aclose(self) -> None:
        return None


class TopDegreeBaseline:
    """Predicts the top-``size`` nodes by total degree (in + out edges).

    Self-loops contribute 0 (they tell us nothing about external structure).
    Tie-break: lexicographic on node_id so output is deterministic.
    """

    name = "top_degree"

    def __init__(self, size: int = 5) -> None:
        if size < 0:
            raise ValueError(f"size must be >= 0, got {size}")
        self._size = size

    async def run(self, inputs: ApproachInputs) -> ApproachResult:
        ids = _all_node_ids(inputs)
        if not ids or self._size == 0:
            return ApproachResult(
                approach=self.name,
                metadata={
                    "strategy": "top_degree",
                    "size": self._size,
                    "n_available": len(ids),
                },
            )

        deg: Counter[str] = Counter()
        valid_ids = set(ids)
        for e in inputs.edges:
            a = e.get("node_id_from")
            b = e.get("node_id_to")
            if not a or not b or a == b:
                continue
            if a in valid_ids:
                deg[a] += 1
            if b in valid_ids:
                deg[b] += 1
        # ranked: degree DESC, then node_id ASC
        ranked = sorted(ids, key=lambda nid: (-deg.get(nid, 0), nid))
        chosen = set(ranked[: self._size])

        nodes_out, edges_out = _select_subgraph(inputs, chosen)
        return ApproachResult(
            approach=self.name,
            nodes=nodes_out,
            edges=edges_out,
            required_node_ids=sorted(chosen),
            useful_node_ids=[],
            metadata={
                "strategy": "top_degree",
                "size": self._size,
                "n_available": len(ids),
                "n_picked": len(chosen),
                "max_degree": max(deg.values()) if deg else 0,
            },
        )

    async def aclose(self) -> None:
        return None


# ---- BM25 (lexical) -------------------------------------------------------

# Tokenizer: splits CamelCase / PascalCase / snake_case into pieces and
# lowercases. Drops single-char tokens (low signal, high noise from variable
# names like "i", "x").
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_TOKEN_RE = re.compile(r"[A-Za-z]+")


def _tokenize_for_bm25(text: str) -> list[str]:
    """Split text into lowercase tokens, splitting CamelCase along the way.

    >>> _tokenize_for_bm25("HashMap of URLParser instances")
    ['hash', 'map', 'of', 'url', 'parser', 'instances']

    Wait — "of" is 2 chars so kept; single-char tokens are dropped.
    """
    spaced = _CAMEL_BOUNDARY.sub(" ", text)
    return [t.lower() for t in _TOKEN_RE.findall(spaced) if len(t) > 1]


class BM25Baseline:
    """Top-K nodes by BM25 score against the query.

    Uses the SAME ``nodes_to_texts`` serialization as the embedding retriever
    in ``src.rag``, so BM25 vs. dense retrieval is an apples-to-apples
    comparison: both see the same per-node text.

    Tokenization splits CamelCase/snake_case to lowercase pieces (so
    ``HashMap`` matches a query of ``hash map``).

    Edge cases:
        * Empty diagram or ``size=0`` → predict nothing.
        * Query has no usable tokens (all 1-char or non-letter) → predict nothing.
        * BM25 returns all-zero scores (typical on N≤2 corpora because of the
          IDF formula) → predict nothing rather than picking arbitrary nodes
          by tie-break.
    """

    name = "bm25"

    def __init__(self, size: int = 5) -> None:
        if size < 0:
            raise ValueError(f"size must be >= 0, got {size}")
        self._size = size

    async def run(self, inputs: ApproachInputs) -> ApproachResult:
        nodes = list(inputs.nodes)
        if not nodes or self._size == 0:
            return ApproachResult(
                approach=self.name,
                metadata={
                    "strategy": "bm25",
                    "size": self._size,
                    "n_available": len(nodes),
                },
            )

        query_tokens = _tokenize_for_bm25(inputs.query or "")
        if not query_tokens:
            return ApproachResult(
                approach=self.name,
                metadata={
                    "strategy": "bm25",
                    "size": self._size,
                    "no_query_tokens": True,
                },
            )

        # Lazy imports — keep the registry loadable even if rank_bm25 isn't
        # installed yet (and avoid coupling unrelated tests to it).
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError(
                "rank_bm25 is not installed. Run `pip install rank-bm25` "
                "or reinstall via `pip install -r requirements.txt`."
            ) from e
        from src.rag.node_to_text import nodes_to_texts

        texts = nodes_to_texts(nodes, inputs.edges)
        corpus = [_tokenize_for_bm25(t) for t in texts]
        # BM25Okapi crashes on empty docs; guard with a placeholder token that
        # is unlikely to appear in any real query.
        corpus = [toks if toks else ["__empty_doc__"] for toks in corpus]

        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)

        if not any(s > 0 for s in scores):
            # All-zero scores: query terms don't appear in any node, or the
            # corpus is too small for BM25Okapi's IDF to be positive. Either
            # way, BM25 has nothing to say — abstain rather than return
            # arbitrary lex-first nodes.
            return ApproachResult(
                approach=self.name,
                metadata={
                    "strategy": "bm25",
                    "size": self._size,
                    "all_zero_scores": True,
                    "n_query_tokens": len(query_tokens),
                },
            )

        # Rank: score DESC, then node_id ASC for deterministic tie-breaks.
        ranked = sorted(
            range(len(nodes)),
            key=lambda i: (-float(scores[i]), nodes[i].get("node_id", "")),
        )
        chosen: list[str] = []
        seen: set[str] = set()
        for idx in ranked:
            if scores[idx] <= 0:
                break  # don't pad the prediction with zero-score nodes
            nid = nodes[idx].get("node_id")
            if isinstance(nid, str) and nid and nid not in seen:
                chosen.append(nid)
                seen.add(nid)
                if len(chosen) >= self._size:
                    break

        if not chosen:
            return ApproachResult(
                approach=self.name,
                metadata={
                    "strategy": "bm25",
                    "size": self._size,
                    "no_positive_scores": True,
                    "n_query_tokens": len(query_tokens),
                },
            )

        nodes_out, edges_out = _select_subgraph(inputs, set(chosen))
        return ApproachResult(
            approach=self.name,
            nodes=nodes_out,
            edges=edges_out,
            required_node_ids=sorted(chosen),
            useful_node_ids=[],
            metadata={
                "strategy": "bm25",
                "size": self._size,
                "n_available": len(nodes),
                "n_picked": len(chosen),
                "n_query_tokens": len(query_tokens),
                "max_score": float(max(scores)),
            },
        )

    async def aclose(self) -> None:
        return None
