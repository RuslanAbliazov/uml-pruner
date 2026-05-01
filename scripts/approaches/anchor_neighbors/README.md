# Approach #2 — Anchor + neighbors + prune

**Status:** implemented.

## Idea

1. **Generate candidates with RAG.**
   The embedding retriever (built by `scripts/build_index.py`) returns the
   top-`n_candidates` classes most similar to the user's query. Default is
   10; configurable via `approaches.anchor_neighbors.n_candidates` in the
   YAML config.
2. **Pick the anchor (LLM).**
   The LLM is asked to pick the SINGLE best anchor out of those candidates.
   The prompt lives in `prompts/anchor_select_{system,user}.txt`. The LLM
   returns the chosen `node_id` plus a short reason.
3. **Expand by neighborhood.**
   Every direct neighbor of the anchor is collected — both outgoing AND
   incoming edges, every relation kind (Inheritance / Association /
   Dependency / others). Self-loops are skipped.
4. **Prune (LLM).**
   The anchor + its neighbors plus the edges between them are sent to the
   LLM, which classifies each node as REQUIRED / USEFUL / IRRELEVANT. The
   prune prompt is in `prompts/anchor_prune_{system,user}.txt`. The anchor
   itself is force-kept (added to REQUIRED if the LLM forgets it).

The result is the standard `{nodes, edges, metadata}` shape consumed by
`src/evaluation/evaluator.py`.

## Why it might beat the baseline (#1)
- The candidate set is **graph-grounded**: every kept class has a structural
  link to the anchor, which curbs LLM hallucinations.
- Smaller LLM input — only the anchor + its 1-hop neighborhood, not 200+
  classes from a coarse RAG batch.

## Why it might lose
- The whole answer hinges on the anchor pick. If RAG misses the right anchor
  among the top-`n_candidates`, recall craters.
- 1-hop expansion may not reach indirectly related classes (test classes,
  helpers two hops away).

## Prerequisites

```bash
# 1. Consolidated dataset
python scripts/build_dataset.py

# 2. Cleaned diagrams
python scripts/normalize_diagrams.py

# 3. Embedding indices for every project (once per model change)
pip install -r requirements-embeddings.txt
python scripts/build_index.py --all
```

## Run

```bash
# Whole dataset
python scripts/approaches/anchor_neighbors/run.py
# Equivalent:
python scripts/benchmark.py --approach anchor_neighbors

# Restrict to a repo / subset / single sample
python scripts/benchmark.py --approach anchor_neighbors --repo apache/hadoop
python scripts/benchmark.py --approach anchor_neighbors --limit 5
python scripts/benchmark.py --approach anchor_neighbors --sample-id <id>
```

Outputs land under `data/results/anchor_neighbors/` (which is git-ignored
via the `data/` rule in `.gitignore`):

- `<sample_id>.json` — pruned subgraph + per-sample metadata (selected
  anchor, candidate list with retrieval scores, neighbor / subgraph counts,
  LLM selection reason).
- `evaluation_report.json` — aggregate metrics, written automatically by the
  benchmark unless `--no-eval` is passed.
- `errors.json` — per-sample failures (missing index, runner errors, etc.).

## Configuration

`configs/config.yaml`:

```yaml
approaches:
  anchor_neighbors:
    n_candidates: 10            # how many candidates the retriever returns
    max_subgraph_nodes: 200     # safety cap on anchor + neighbors size
                                # (-1 / null = disable cap)
```

The retrieval model and cache dir are read from the shared `embeddings:`
block (same one used by `build_index.py`).

## Source
- Runner: `src/approaches/anchor_neighbors/runner.py`
- Prompts: `prompts/anchor_select_{system,user}.txt`,
  `prompts/anchor_prune_{system,user}.txt`
