# Approach #2 — Anchor + neighbors + prune

**Status:** implemented.

## Idea

1. **Generate candidates with RAG.**
   The embedding retriever returns the top-`n_candidates` classes most
   similar to the user's query. Default is 10; configurable via
   `approaches.anchor_neighbors.n_candidates` in `configs/config.yaml`.
2. **Pick the anchor.** Two interchangeable strategies, selected by the
   `approaches.anchor_neighbors.anchor_selector` key in the YAML:
   - `"llm"` (default) — the LLM is asked to pick the single best anchor
     out of the candidates. See `prompts/select_system.txt`,
     `prompts/select_user.txt`.
   - `"reranker"` — a cross-encoder (configured under the top-level
     `reranker:` section) scores every candidate against the query, and
     the top-1 becomes the anchor. No LLM call on this stage.
3. **Expand by neighborhood.**
   Every direct neighbor of the anchor is collected — both outgoing AND
   incoming edges, every relation kind (Inheritance / Association /
   Dependency / others). Self-loops are skipped.
4. **Prune (LLM).**
   The anchor + its neighbors plus the edges between them are sent to the
   LLM, which classifies each node as REQUIRED / USEFUL / IRRELEVANT
   (`prompts/prune_system.txt`, `prompts/prune_user.txt`). The anchor
   itself is force-kept.

The result is the standard `{nodes, edges, metadata}` shape consumed by
`src/eval/evaluator.py`.

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

The approach has its own CLI right next to the source:

```bash
# Whole dataset
python src/approaches/anchor_neighbors/run.py

# Restrict to a repo / subset / single sample
python src/approaches/anchor_neighbors/run.py --repo apache/hadoop
python src/approaches/anchor_neighbors/run.py --limit 5
python src/approaches/anchor_neighbors/run.py --sample-id <id>

# Skip eval
python src/approaches/anchor_neighbors/run.py --no-eval
```

The generic ``scripts/run.py --approach anchor_neighbors`` works too — both
share the same factory and produce identical output.

Outputs go to `data/results/anchor_neighbors/` (gitignored):

- `<sample_id>.json` — pruned subgraph + per-sample metadata (selected
  anchor, candidate list with retrieval scores, neighbor / subgraph counts,
  LLM selection reason).
- `evaluation_report.json` — aggregate metrics (auto-written unless
  `--no-eval` is passed).
- `errors.json` — per-sample failures.

## Configuration

`configs/config.yaml`:

```yaml
approaches:
  anchor_neighbors:
    n_candidates: 10            # how many candidates the retriever returns
    max_subgraph_nodes: 200     # safety cap on anchor + neighbors size
                                # (-1 / null = disable cap)
    anchor_selector: "llm"      # "llm" | "reranker" — which engine picks
                                # the single anchor on stage 2
    outputs_dir: "data/results/anchor_neighbors"
    llm_traces_dir: "data/llm_traces/anchor_neighbors"

# Used only when anchor_selector == "reranker":
reranker:
  model: "BAAI/bge-reranker-v2-m3"
  device: "auto"
  batch_size: 16
  max_seq_length: 512           # null / -1 => use model default
```

The retrieval model and cache dir are read from the shared `embeddings:`
block (same one used by `scripts/build_index.py`).

The selector name is automatically appended to `outputs_dir` and
`llm_traces_dir` as a subfolder, so two runs (one with `"llm"`, one with
`"reranker"`) land side-by-side and never overwrite each other:

```
data/results/anchor_neighbors/
├── llm/
│   ├── samples/...
│   ├── report.jsonl
│   └── aggregate.json
└── reranker/
    ├── samples/...
    ├── report.jsonl
    └── aggregate.json
```

### Comparing the two anchor selectors

```bash
# 1) Run the LLM-based selector (current behaviour).
#    Set `anchor_selector: "llm"` in the YAML, then:
python src/approaches/anchor_neighbors/run.py --limit 50

# 2) Switch the YAML key to `anchor_selector: "reranker"` and re-run:
python src/approaches/anchor_neighbors/run.py --limit 50

# 3) Diff the aggregate metrics:
diff data/results/anchor_neighbors/llm/aggregate.json \
     data/results/anchor_neighbors/reranker/aggregate.json
```

The shape of `aggregate.json` is identical for both runs (the `anchor`
stage exposes the same fields — `anchor_in_required_rate`, `mean_recall`,
etc.), so the two files are directly comparable.

## Source layout

The pipeline is split across small, single-purpose files so each stage is
easy to read in isolation:

```
anchor_neighbors/
├── run.py                       # CLI entry point — runs the approach over the dataset
├── pipeline.py                  # Orchestrator: glues the four stages into ApproachResult
├── settings.py                  # Reads YAML; build_runner factory
├── stage_outputs.py             # Common StageOutcome contract
├── stage1_retrieve.py           # Stage 1 — RAG retrieval (CandidateRetriever)
├── stage2_select_anchor.py      # Stage 2a — LLM picks one anchor (with fallback)
├── stage2_rerank_anchor.py      # Stage 2b — cross-encoder picks top-1 anchor
├── stage3_expand_neighbors.py   # Stage 3 — collect neighborhood + cap
├── stage4_prune.py              # Stage 4 — LLM REQUIRED / USEFUL / IRRELEVANT
├── prompt_templates.py          # Tiny wrappers over ./prompts/*.txt
├── llm_trace.py                 # Per-stage last-call request/response dump
├── ground_truth.py              # Loads gold labels by (repo, query)
├── metrics.py                   # Per-stage quality metrics vs. gold
├── debug_report.py              # JSONL writer + aggregation
├── prompts/
│   ├── select_{system,user}.txt
│   └── prune_{system,user}.txt
└── README.md
```
