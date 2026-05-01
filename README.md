# UML Pruner

A tool for pruning large UML class diagrams down to the subset of nodes and
edges that are relevant to a natural-language user query. Given a full UML
diagram (e.g. of a sizable Java codebase) and a query like "Show the main and
supporting classes involved in testing the combined merging of function tags
and listings", the tool returns a smaller diagram containing only the
classes/interfaces and relations the user actually needs to understand.

## What the tool does

Inputs:
- A user query in natural language.
- A full UML diagram in JSON form (see `uml.json` for the schema/example).

Output:
- A pruned diagram in the same `{nodes, edges}` JSON shape, plus a `metadata`
  block with diagnostic info (original/filtered counts, which IDs are
  REQUIRED vs USEFUL, which Stage 1 strategy was used, etc).

### UML JSON schema (see `uml.json`)

```json
{
  "nodes": [
    {
      "type": "class | interface | ...",
      "name": "fully.qualified.Name",
      "node_id": "fully.qualified.Name",
      "methods": ["public foo(...)", "..."],
      "description": "free-text class description"
    }
  ],
  "edges": [
    {
      "node_id_from": "...",
      "node_id_to": "...",
      "description": "Inheritance | Association | Dependency | ...",
      "subdescription": "Implements | HAS_MANY | PARAMETER_TYPE | ...",
      "label": "optional"
    }
  ]
}
```

`node_id` is the canonical identifier used everywhere (annotations, results,
embedding indices). It is typically the fully-qualified class/interface name.

## High-level pipeline

The pruning pipeline has **two stages**, both protected by an autosplit
driver that recursively splits batches if a prompt would overflow the LLM
context window.

### Stage 1 — coarse filter (package level)

Two interchangeable strategies; controlled via config / CLI flags:

1. **LLM Stage 1 (default).** Classes are grouped by Java-style package, then
   the LLM is asked, batch by batch, which packages are likely to contain
   classes relevant to the query. Output: a set of surviving `node_id`s.
   Implemented in `src/pipeline/stage1_coarse.py`.
2. **Embedding retrieval (opt-in).** A pre-built local vector index of the
   diagram is loaded; the query is embedded; the top-K most similar nodes are
   kept. No LLM calls. Falls back to LLM Stage 1 if the index is missing,
   stale, or returns 0 hits. Implemented in `src/embeddings/*` and wired up in
   `src/pipeline/pipeline.py:_stage1_via_embeddings`.

### Stage 2 — fine filter (class level)

The surviving classes plus their incident edges are batched and sent to the
LLM, which classifies each class as one of `REQUIRED`, `USEFUL`, or
`IRRELEVANT`. Only `REQUIRED` and `USEFUL` are kept. Implemented in
`src/pipeline/stage2_midlevel.py`.

The final pruned subgraph is built by `src/preprocessing/compressor.py:filter_subgraph`,
which keeps only the surviving nodes and edges whose endpoints both survived.

### Autosplit / token budget

`src/llm/budget.py` (`TokenBudget`) tracks the LLM context window, output
reserve, and a safety margin. `src/pipeline/autosplit.py` wraps each stage's
LLM call so that any batch which won't fit is recursively split in half (up
to `max_split_depth`). This makes both stages safe for very large diagrams.

## Approaches (benchmark-able)

The project hosts several interchangeable strategies for producing a pruned
diagram. They share the same input/output contract
(`src/approaches/base.py::ApproachRunner`) and can be benchmarked uniformly
via `scripts/benchmark.py`.

| #   | Name                  | Status      | Idea                                            |
|-----|-----------------------|-------------|-------------------------------------------------|
| 1   | `rag_classes_filter`  | implemented | RAG batch (50–200 cls) → LLM REQUIRED/USEFUL    |
| 2   | `anchor_neighbors`    | stub        | Pick anchors → expand neighborhood → LLM prune  |
| 3   | `agentic_chunks`      | stub        | Chunk diagram → per-chunk LLM survey → synth    |
| 4   | `human_like_agent`    | stub        | Anchors → expand by betweenness/calls → prune   |

Each approach has its own user-facing README and runner under
`scripts/approaches/<name>/`.

## Repository layout

```
uml-pruner/
├── README.md                     # this file
├── uml.json                      # tiny example diagram (Observer pattern)
├── requirements.txt              # core deps (LLM pipeline only)
├── requirements-embeddings.txt   # extra deps for local embedding retrieval
├── annotations.csv               # raw human annotations (multi-annotator)
│
├── configs/
│   ├── config.yaml               # main pipeline config (LLM, stages, embeddings, paths)
│   └── examples/                 # extra example configs
│
├── prompts/
│   ├── stage1_system.txt         # Stage 1 system prompt (package-level filter)
│   ├── stage1_user.txt           # Stage 1 user-prompt template
│   ├── stage2_system.txt         # Stage 2 system prompt (REQUIRED/USEFUL classifier)
│   └── stage2_user.txt           # Stage 2 user-prompt template
│
├── uml_with_methods/             # source diagrams (full, untouched)
├── full_diagrams_fixed_generic/  # legacy diagrams (kept for backwards compat)
│
├── src/
│   ├── approaches/               # pluggable UML-pruning strategies
│   │   ├── base.py               # ApproachRunner protocol + ApproachInputs/Result
│   │   ├── __init__.py           # registry of approaches
│   │   ├── rag_classes_filter/   # #1 — wraps the legacy 2-stage pipeline
│   │   ├── anchor_neighbors/     # #2 — anchor + neighbors + prune
│   │   ├── agentic_chunks/       # #3 — chunked agentic selection
│   │   └── human_like_agent/     # #4 — anchors + centrality expansion
│   │
│   ├── pipeline/                 # legacy 2-stage pipeline (used by approach #1)
│   │   ├── pipeline.py           # run_pipeline(query, diagram, client, cfg)
│   │   ├── stage1_coarse.py      # LLM Stage 1: package-level coarse filter
│   │   ├── stage2_midlevel.py    # LLM Stage 2: per-class REQUIRED/USEFUL classifier
│   │   └── autosplit.py          # generic recursive batch-splitter for LLM calls
│   │
│   ├── llm/                      # OpenAI-compatible client, prompts, parser, budget
│   ├── preprocessing/            # package grouping, batching, subgraph compressor
│   ├── embeddings/               # local embedding retriever
│   │   ├── encoder.py            # LocalEncoder (sentence-transformers wrapper)
│   │   ├── node_to_text.py       # GRAPH-AWARE text serialization of a node
│   │   ├── cache.py              # on-disk index format + validity checks
│   │   └── retriever.py          # top-K cosine-similarity retrieval
│   │
│   ├── evaluation/
│   │   ├── annotations.py        # loads the consolidated dataset (data/dataset.csv)
│   │   ├── metrics.py            # precision / recall_required / recall_useful / f1
│   │   └── evaluator.py          # evaluates a results dir against the dataset
│   │
│   └── utils/                    # config, io, logger, token counter
│
├── scripts/
│   ├── build_dataset.py          # raw annotations -> consolidated dataset (votes, repo lookup)
│   ├── normalize_diagrams.py     # uml_with_methods/ -> data/diagrams_normalized/
│   ├── build_index.py            # build embedding indices for normalized diagrams
│   ├── benchmark.py              # GENERIC: run any approach on the dataset, evaluate
│   ├── evaluate.py               # evaluate an arbitrary results-dir vs the dataset
│   ├── eval_retriever.py         # evaluate embedding retrieval in isolation
│   ├── retrieve.py               # query an existing embedding index (debug)
│   ├── batch_process.py          # legacy CLI: run the 2-stage pipeline (= approach #1)
│   ├── run_pipeline.py           # legacy CLI: prune ONE diagram with ONE query
│   │
│   └── approaches/               # per-approach run scripts + READMEs
│       ├── rag_classes_filter/   # README + run.py (delegates to benchmark.py)
│       ├── anchor_neighbors/     # README + run.py (stub)
│       ├── agentic_chunks/       # README + run.py (stub)
│       └── human_like_agent/     # README + run.py (stub)
│
├── tests/
│   ├── unit/                     # tests for autosplit, budget, compressor, metrics
│   └── integration/              # mocked end-to-end pipeline test
│
├── data/
│   ├── dataset.csv               # consolidated dataset (built by build_dataset.py)
│   ├── diagrams_normalized/      # cleaned diagrams (built by normalize_diagrams.py)
│   ├── embeddings/               # cached embedding indices, keyed by diagram stem
│   └── results/                  # per-approach result JSONs + eval reports
│
└── logs/                         # log files (configurable in config.yaml)
```

## Configuration (`configs/config.yaml`)

Key sections:
- `llm` — provider, `base_url`, `api_key` (supports `${ENV_VAR}`), model,
  temperature, retries, and **context budget** (`context_window`,
  `output_reserve`, `safety_margin`). The client is OpenAI-API compatible
  (works with OpenAI, OpenRouter, Together, Ollama/LM Studio, etc).
- `pipeline.stage1` — batch size, parallelism, relevance levels.
- `pipeline.stage2` — batch size, parallelism, output categories,
  `max_output_classes`.
- `pipeline.max_split_depth` — autosplit recursion cap (shared by both stages).
- `embeddings` — opt-in semantic retrieval as a Stage 1 replacement: model,
  device (`auto|cuda|mps|cpu`), `top_k`, cache dir, text-build limits.
- `evaluation.metrics` — which metrics the evaluator computes.
- `paths` — diagrams dir, annotations file, results dir, logs dir, prompts dir.

## CLI usage

### One-time data preparation

```bash
# 1. Build the consolidated dataset (votes per node, repo lookup, etc.).
#    Default output: data/dataset.csv
python scripts/build_dataset.py

# 2. Normalize diagrams for RAG (drop noisy edge fields, dedup).
#    uml_with_methods/  ->  data/diagrams_normalized/
python scripts/normalize_diagrams.py

# 3. Build embedding indices (optional, only if you'll use approach #1
#    with the embedding retriever, or #2 / #4 once implemented).
pip install -r requirements-embeddings.txt
python scripts/build_index.py --all
```

### Run an approach end-to-end (generation + evaluation)

```bash
# List all registered approaches
python scripts/benchmark.py --list

# Run baseline on the whole dataset
python scripts/benchmark.py --approach rag_classes_filter

# Or use the per-approach wrapper (same effect; lives next to its README)
python scripts/approaches/rag_classes_filter/run.py

# Restrict to one repo / one sample / a few samples
python scripts/benchmark.py --approach rag_classes_filter --repo apache/hadoop
python scripts/benchmark.py --approach rag_classes_filter --sample-id <id>
python scripts/benchmark.py --approach rag_classes_filter --limit 5
```

Outputs land under `data/results/<approach>/`:
- `<sample_id>.json` — pruned subgraph per sample (`{nodes, edges, metadata}`).
- `evaluation_report.json` — aggregate metrics, written automatically unless
  `--no-eval` is passed.
- `errors.json` — per-sample failures (missing diagram, runner error, etc.).

### Evaluate an existing result dir

```bash
python scripts/evaluate.py \
  --dataset data/dataset.csv \
  --results-dir data/results/rag_classes_filter \
  --output      data/results/rag_classes_filter/evaluation_report.json
```

### Legacy single-diagram CLI

```bash
python scripts/run_pipeline.py \
  --query "Show classes responsible for X" \
  --diagram data/diagrams_normalized/disruptor.json \
  --output  data/results/disruptor_X.json
```

## Evaluation methodology

`annotations.csv` is the raw multi-annotator file. The benchmarks evaluate
against the **consolidated dataset** built from it by
`scripts/build_dataset.py`:

- One row per `(sample_id, central_node, query)`.
- `entity_annotations`: voted map `node_id → "required" | "useful"` (the
  build script applies a strict-majority vote per node, with a
  `required > useful > irrelevant` priority for ties; `irrelevant` is dropped
  from the final map).
- `repo` is resolved by searching for `central_node` inside each diagram's
  JSON (no prefix tables).
- Only `Finalized` annotations are used; samples with fewer than two
  annotators are dropped by default (`--keep-single` to override).

`src/evaluation/metrics.py` computes precision, `recall_required`,
`recall_useful`, overall recall, and F1, comparing the approach's
`required_node_ids ∪ useful_node_ids` against the dataset.

## Notes for future agents

- The example `uml.json` at the repo root is a tiny Observer-pattern diagram,
  illustrating the expected JSON shape; real diagrams live in
  `uml_with_methods/` (and the cleaned form in `data/diagrams_normalized/`)
  and can have thousands of nodes.
- All data preparation is centralized in `scripts/build_dataset.py` and
  `scripts/normalize_diagrams.py`. Other scripts treat their outputs as
  immutable.
- Approaches are pluggable: implement `src/approaches/<name>/runner.py`,
  register it in `src/approaches/__init__.py::REGISTRY`, drop a README +
  `run.py` into `scripts/approaches/<name>/`, and you can benchmark it via
  `scripts/benchmark.py --approach <name>`.
- The pipeline is async (`asyncio`); LLM calls are parallelized per stage
  with `max_parallel_requests`.
- All LLM responses are constrained to JSON; `src/llm/parser.py` is tolerant
  of stray prose / fences from less obedient models.
