# Approach #1 — RAG batch + LLM filter (baseline)

**Status:** implemented (this is the project's existing 2-stage pipeline,
re-exposed under the unified approach interface).

## Idea

1. **Stage 1 — candidate batch (50–200 classes).**
   Either:
   - **Embedding retrieval** (preferred): take the top-K classes whose graph-aware
     text embeddings are most similar to the query.
   - **LLM package filter** (fallback): group classes by Java package, ask the
     LLM which packages look relevant, keep their classes.
2. **Stage 2 — LLM classifier.**
   Send the candidate batch (with neighborhood context) to the LLM. For each
   class, classify it as `REQUIRED`, `USEFUL`, or `IRRELEVANT`. Drop the
   `IRRELEVANT` ones.

The final pruned subgraph is `{required ∪ useful}` plus the edges between them.

## Strengths
- Mature implementation; works on diagrams up to ~20k nodes via autosplit.
- Embedding stage cuts LLM cost ~10×.

## Weaknesses
- Stage 1 has a hard recall ceiling: anything the retriever / package filter
  drops is lost forever (Stage 2 never sees it).
- Stage 2 can hallucinate a class as `REQUIRED` even if it has no structural
  connection to the rest of the answer.

## Run

```bash
# Build prerequisites once
python scripts/build_dataset.py
python scripts/normalize_diagrams.py
python scripts/build_index.py --all

# Benchmark (writes to data/results/rag_classes_filter/)
python scripts/approaches/rag_classes_filter/run.py
# Equivalent to:
python scripts/benchmark.py --approach rag_classes_filter
```

## Output
- `data/results/rag_classes_filter/<sample_id>.json` — pruned subgraph per
  sample (standard `{nodes, edges, metadata}` shape).
- `data/results/rag_classes_filter/evaluation_report.json` — aggregate metrics
  written after the benchmark finishes.

## Source
- Runner adapter: `src/approaches/rag_classes_filter/runner.py`
- Underlying pipeline: `src/pipeline/pipeline.py`
  (`run_pipeline()`, Stage 1: `stage1_coarse.py`, Stage 2: `stage2_midlevel.py`)
