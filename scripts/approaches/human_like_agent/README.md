# Approach #4 — Human-like agent

**Status:** stub. Public interface defined; implementation TBD.

## Idea

Mimic how a real human investigator explores an unfamiliar codebase looking
for the answer to a query.

1. **Find anchors (small N).**
   The agent reads the query and asks: _"which N classes are most likely
   the entry points / focus of this question?"_. We pick them via the
   embedding retriever, optionally with an LLM rerank step.
2. **Expand by structural importance.**
   For every anchor, surface neighbors that are *structurally important* in
   the codebase — using the same signals the human annotators used:
   - **betweenness centrality** (`bc_threshold`),
   - **call_in_code** count (`calls_threshold`).
   These are the classes a careful reader would look at next, because they
   sit on the natural call/data paths through the system.
3. **Prune.**
   The agent inspects the expanded neighborhood and drops classes that look
   incidental, keeping the rest as REQUIRED / USEFUL.

## Why this might beat the others
- The dataset itself was annotated using a slider over `bc_threshold` and
  `calls_threshold` (see `slider_state` in `annotations.csv`). Leaning on the
  same signals should align well with the annotators' mental model.
- Composable: anchor stage, expansion stage and prune stage can each be
  ablated and benchmarked independently.

## Why it might lose
- Centrality scores require precomputation per repo. They depend on the
  global graph structure, so they'll be a separate one-time build step.
- For very small/tightly-coupled diagrams, betweenness saturates and stops
  discriminating.

## Plan / TODOs
- [ ] Precompute centrality metrics per diagram (script:
      `scripts/precompute_centrality.py`). Cache to
      `data/centrality/<diagram_stem>.json`.
- [ ] Implement `_pick_anchors()` (RAG + optional LLM rerank).
- [ ] Implement `_expand_by_centrality()` using cached metrics.
- [ ] Implement `_prune()` (LLM call; reuse Stage 2 prompt).
- [ ] YAML config: `approaches.human_like_agent.*`.

## Run (once implemented)

```bash
# One-time precomputation
python scripts/precompute_centrality.py --all

# Benchmark
python scripts/approaches/human_like_agent/run.py
# Equivalent to:
python scripts/benchmark.py --approach human_like_agent
```

## Source
- `src/approaches/human_like_agent/runner.py`
