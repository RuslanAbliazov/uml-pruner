# Approach #1 — RAG batch + LLM classifier (baseline)

**Status:** implemented.

## Idea

1. **Stage 1 — coarse filter.** Either a package-level LLM filter or
   embedding retrieval (top-K) returns 50–500 candidate classes.
2. **Stage 2 — class-level classifier.** The LLM tags each candidate as
   `REQUIRED`, `USEFUL`, or `IRRELEVANT`.

Both stages are protected by an autosplit driver
(`autosplit.py`) that recursively halves any batch whose prompt would
overflow the LLM context window.

## Run

```bash
python scripts/run.py --approach rag_classes_filter
python scripts/run.py --approach rag_classes_filter --limit 5
python scripts/run.py --approach rag_classes_filter --repo apache/hadoop
```

## Configuration

`configs/config.yaml`:

```yaml
pipeline:
  stage1:
    package_batch_size: 20
    max_parallel_requests: 5
  stage2:
    class_batch_size: 120
    max_parallel_requests: 3
    max_output_classes: 50
  max_split_depth: 8

embeddings:
  enabled: false      # flip on to replace LLM Stage 1 with semantic retrieval
  top_k: 500
```

## Source layout
- `runner.py`           — `ApproachRunner` adapter wiring inputs/outputs
- `pipeline.py`         — orchestrates Stage 1 + Stage 2
- `stage1.py`           — package-level coarse filter (LLM)
- `stage2.py`           — REQUIRED/USEFUL classifier (LLM)
- `autosplit.py`        — recursive batch splitter
- `prompts/`            — stage prompts owned by this approach
