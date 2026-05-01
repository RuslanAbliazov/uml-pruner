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
- A full UML diagram in JSON form.

Output:
- A pruned diagram in the same `{nodes, edges}` JSON shape, plus a `metadata`
  block with diagnostic info (original/filtered counts, which IDs are
  REQUIRED vs USEFUL, per-approach diagnostics, etc).

### UML JSON schema

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

## Approaches

Every approach implements the `ApproachRunner` protocol
(`src/core/types.py`) and lives in its own self-contained package under
`src/approaches/<name>/`. Each package owns its runner, prompts, and README.

| #   | Name                  | Status      | Idea                                            |
|-----|-----------------------|-------------|-------------------------------------------------|
| 1   | `rag_classes_filter`  | implemented | RAG batch (50–200 cls) → LLM REQUIRED/USEFUL    |
| 2   | `anchor_neighbors`    | implemented | RAG candidates → LLM picks anchor → 1-hop expand → LLM prune |
| 3   | `agentic_chunks`      | stub        | Chunk diagram → per-chunk LLM survey → synth    |
| 4   | `human_like_agent`    | stub        | Anchors → expand by betweenness/calls → prune   |

The factory and registry live in `src/approaches/__init__.py`.

### Approach #1 — `rag_classes_filter`

Two stages, both protected by an autosplit driver that recursively splits
batches if a prompt would overflow the LLM context window.

1. **Stage 1 — coarse filter (package level).** Either
   - LLM Stage 1: classes are grouped by package; the LLM picks relevant
     packages batch by batch
     (`src/approaches/rag_classes_filter/stage1.py`); or
   - Embedding retrieval (opt-in): a pre-built local vector index returns
     the top-K most similar nodes (no LLM calls).
2. **Stage 2 — fine filter (class level).** Surviving classes plus their
   incident edges are batched and sent to the LLM, which classifies each
   class as `REQUIRED`, `USEFUL`, or `IRRELEVANT`
   (`src/approaches/rag_classes_filter/stage2.py`).

The final pruned subgraph is built by
`src/approaches/_common/compressor.py:filter_subgraph`.

### Approach #2 — `anchor_neighbors`

1. RAG returns top-`n_candidates` classes for the query.
2. The LLM picks ONE of them as the **anchor**.
3. All direct neighbors (any direction, any relation kind) are collected.
4. The LLM classifies the anchor + neighborhood as REQUIRED / USEFUL /
   IRRELEVANT. The anchor itself is force-kept.

See `src/approaches/anchor_neighbors/README.md` for details.

### Token budget

`src/llm/budget.py::TokenBudget` tracks the LLM context window, output
reserve, and a safety margin.
`src/approaches/rag_classes_filter/autosplit.py` wraps each stage's LLM
call so any batch which won't fit is recursively split in half (up to
`max_split_depth`). This keeps the baseline pipeline safe for very large
diagrams.

## Repository layout

```
uml-pruner/
├── README.md
├── requirements.txt              # core deps (LLM pipeline only)
├── requirements-embeddings.txt   # extra deps for local embedding retrieval
├── annotations.csv               # raw human annotations (multi-annotator)
├── uml_with_methods/             # source diagrams (gitignored, 8 .json files)
│
├── configs/
│   ├── config.yaml               # main config (LLM, stages, embeddings, paths)
│   └── examples/                 # provider-specific examples
│
├── data/                         # gitignored: everything generated
│   ├── dataset.csv               # consolidated dataset
│   ├── diagrams_normalized/      # normalized diagrams for RAG
│   ├── embeddings/               # cached embedding indices
│   └── results/<approach>/       # per-approach run outputs + reports
│
├── scripts/                      # ENTRY POINTS — only thin CLI wrappers
│   ├── normalize_diagrams.py     # uml_with_methods/ -> data/diagrams_normalized/
│   ├── build_dataset.py          # annotations.csv  -> data/dataset.csv
│   ├── build_index.py            # build embedding indices
│   ├── retrieve.py               # RAG: query an index (debug)
│   ├── eval_retriever.py         # RAG: recall against the dataset
│   ├── run.py                    # GENERIC: run any approach over the dataset
│   └── eval.py                   # evaluate an existing results dir
│
├── src/
│   ├── core/                     # shared primitives
│   │   ├── io.py                 # load/save diagrams + JSON helpers
│   │   ├── config.py             # YAML loader (with ${ENV_VAR} expansion)
│   │   ├── logger.py
│   │   ├── tokens.py             # token-count helper
│   │   └── types.py              # ApproachInputs / ApproachResult / ApproachRunner
│   │
│   ├── llm/
│   │   ├── client.py             # OpenAI-compatible async client
│   │   ├── parser.py             # tolerant JSON parser
│   │   ├── budget.py             # TokenBudget
│   │   └── prompt_loader.py      # load_prompt(path), render_prompt(path, **vars)
│   │
│   ├── rag/                      # local embedding retrieval
│   │   ├── encoder.py            # LocalEncoder (sentence-transformers wrapper)
│   │   ├── node_to_text.py       # graph-aware text serialization of a node
│   │   ├── cache.py              # on-disk index format + validity checks
│   │   └── retriever.py          # top-K cosine-similarity retrieval
│   │
│   ├── eval/
│   │   ├── annotations.py        # loads the consolidated dataset
│   │   ├── metrics.py            # precision / recall_required / recall_useful / f1
│   │   └── evaluator.py          # evaluate a results dir against the dataset
│   │
│   └── approaches/
│       ├── __init__.py           # registry of approaches
│       ├── _common/              # cross-approach helpers
│       │   ├── compressor.py     # filter_subgraph + class representation builders
│       │   ├── package_grouper.py
│       │   └── batching.py
│       │
│       ├── rag_classes_filter/   # #1 — implemented
│       │   ├── runner.py         # ApproachRunner adapter
│       │   ├── pipeline.py       # run_pipeline(query, diagram, client, cfg)
│       │   ├── stage1.py
│       │   ├── stage2.py
│       │   ├── autosplit.py
│       │   ├── prompts/          # owned by THIS approach
│       │   └── README.md
│       │
│       ├── anchor_neighbors/     # #2 — implemented
│       │   ├── runner.py
│       │   ├── prompts/
│       │   │   ├── select_{system,user}.txt
│       │   │   └── prune_{system,user}.txt
│       │   └── README.md
│       │
│       ├── agentic_chunks/       # #3 — stub
│       └── human_like_agent/     # #4 — stub
│
├── tests/
│   ├── unit/                     # autosplit, budget, compressor, metrics
│   └── integration/              # mocked end-to-end pipeline test
│
└── logs/                         # log files (configurable in config.yaml)
```

### Two flat layers, not three

We deliberately **don't** split between `scripts/` and `src/<feature>/cli/`.
The convention is:

- `scripts/*.py` are thin entry points: argparse + glue.
- `src/<package>/` holds the actual logic and exports importable functions.

Approaches are fully self-contained: a single `src/approaches/<name>/`
folder holds the runner, the prompts, and the README. To understand or
modify an approach you only need to read that one folder.

## Configuration (`configs/config.yaml`)

Key sections:
- `llm` — provider, `base_url`, `api_key` (supports `${ENV_VAR}`), model,
  temperature, retries, and **context budget** (`context_window`,
  `output_reserve`, `safety_margin`). The client is OpenAI-API compatible
  (works with OpenAI, OpenRouter, Together, Ollama/LM Studio, etc).
- `pipeline.stage1` / `pipeline.stage2` — knobs for approach #1
  (batch size, parallelism, output categories).
- `pipeline.max_split_depth` — autosplit recursion cap.
- `embeddings` — RAG model, device, `top_k`, cache dir, text-build limits.
- `approaches.<name>` — per-approach knobs (e.g.
  `approaches.anchor_neighbors.n_candidates`). Absent sections fall back to
  hard-coded defaults inside each runner.
- `evaluation.metrics` — which metrics the evaluator reports.
- `paths` — locations of source diagrams, normalized diagrams, dataset CSV,
  results dir, logs dir.

## CLI usage

### One-time data preparation

```bash
# 1. Build the consolidated dataset (votes per node, repo lookup).
#    Default output: data/dataset.csv
python scripts/build_dataset.py

# 2. Normalize diagrams for RAG (drop noisy edge fields, dedup).
#    uml_with_methods/  ->  data/diagrams_normalized/
python scripts/normalize_diagrams.py

# 3. Build embedding indices (needed for any approach using RAG).
pip install -r requirements-embeddings.txt
python scripts/build_index.py --all
```

### Run an approach end-to-end (generation + evaluation)

```bash
# List all registered approaches
python scripts/run.py --list

# Run on the whole dataset, then evaluate
python scripts/run.py --approach rag_classes_filter
python scripts/run.py --approach anchor_neighbors

# Restrict to one repo / one sample / a few samples
python scripts/run.py --approach anchor_neighbors --repo apache/hadoop
python scripts/run.py --approach anchor_neighbors --sample-id <id>
python scripts/run.py --approach anchor_neighbors --limit 5

# Skip the evaluation step
python scripts/run.py --approach anchor_neighbors --no-eval
```

Outputs land under `data/results/<approach>/`:
- `<sample_id>.json` — pruned subgraph per sample (`{nodes, edges, metadata}`).
- `evaluation_report.json` — aggregate metrics (auto-written unless `--no-eval`).
- `errors.json` — per-sample failures (missing diagram, runner error, etc.).

### Evaluate an existing result dir

```bash
python scripts/eval.py \
  --dataset data/dataset.csv \
  --results-dir data/results/anchor_neighbors \
  --output      data/results/anchor_neighbors/evaluation_report.json
```

### Test the RAG retriever in isolation

```bash
# Run a single query against a built index
python scripts/retrieve.py \
  --diagram uml_with_methods/ghidra.json \
  --query "Show classes responsible for defining external locations" \
  --top-k 50 \
  --output data/results/ghidra_retrieved.json

# Sweep top-K and report recall_required / recall_useful on the dataset
python scripts/eval_retriever.py --top-k 100 300 500
python scripts/eval_retriever.py --repo NationalSecurityAgency/ghidra --show-misses
```

## Evaluation methodology

`annotations.csv` is the raw multi-annotator file. Benchmarks evaluate
against the **consolidated dataset** built by `scripts/build_dataset.py`:

- One row per `(sample_id, central_node, query)`.
- `entity_annotations`: voted map `node_id → "required" | "useful"` (strict
  majority vote per node, with a `required > useful > irrelevant` priority
  for ties; `irrelevant` is dropped from the final map).
- `repo` is resolved by searching for `central_node` inside each diagram's
  JSON (no prefix tables).
- Only `Finalized` annotations are used; samples with fewer than two
  annotators are dropped by default (`--keep-single` to override).

`src/eval/metrics.py` computes precision, `recall_required`,
`recall_useful`, overall recall, and F1, comparing the approach's
`required_node_ids ∪ useful_node_ids` against the dataset.

### Ground-truth isolation

`ApproachInputs` deliberately omits `central_node` and the per-sample
annotations. Approaches see ONLY the user query and the full normalized
diagram — same information as in production. `sample_id` and `repo` are
passed in for output filenames and on-disk index lookup, and runners must
not derive any decision from them.

## Notes for future agents

- All data preparation is centralized in `scripts/build_dataset.py` and
  `scripts/normalize_diagrams.py`. Other scripts treat their outputs as
  immutable.
- Approaches are pluggable: implement `src/approaches/<name>/runner.py`,
  register it in `src/approaches/__init__.py::REGISTRY`, drop prompts into
  `src/approaches/<name>/prompts/`, and you can benchmark it via
  `scripts/run.py --approach <name>`.
- All LLM calls are async; per-stage parallelism is controlled by
  `max_parallel_requests` in the YAML.
- LLM responses are constrained to JSON; `src/llm/parser.py` is tolerant of
  stray prose / fences from less obedient models.
