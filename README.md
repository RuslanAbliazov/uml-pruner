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

## Baselines and oracles

Lightweight reference points for any "real" approach to be compared
against. None of the query-agnostic baselines load any model; `bm25` pulls
in `rank_bm25` (small, pure-Python). All runnable in CI without external
services.

Why include these in a serious benchmark: claims like "our approach reaches
F1 = 0.40" are only interpretable next to a floor and ceiling. If
`random_subset` of the same size already reaches F1 = 0.35, the "real"
approach is contributing very little. If `bm25` matches the dense
retriever's score on this dataset, the embedding stack is not earning its
keep. If `oracle_central_plus_neighbors` only reaches F1 = 0.55, no "1
anchor + 1-hop" architecture can go higher and the constraint is structural.

### Query-agnostic baselines

Registered alongside the real approaches; runnable via `scripts/run.py
--approach <name>`.

| Name           | What it predicts                                  |
|----------------|---------------------------------------------------|
| `empty`        | nothing (sanity floor: F1 = 0)                    |
| `full_diagram` | the entire diagram (recall = 1.0; precision tiny) |
| `random_subset`| `size` node_ids picked uniformly at random; seeded per `sample_id` (reproducible) |
| `top_degree`   | top-`size` nodes by total degree (in + out); query-agnostic graph centrality |

### Lexical baseline

| Name   | What it predicts                                                               |
|--------|--------------------------------------------------------------------------------|
| `bm25` | top-`size` nodes by BM25 against the query, over the SAME node-text serialization (`src.rag.node_to_text.nodes_to_texts`) the embedding retriever uses. |

The tokenizer splits CamelCase / snake_case to lowercase pieces and drops
single-char tokens (so `HashMapImpl` matches a query of `hash map`). When
the query contributes no usable tokens, or BM25 returns all-zero scores
(as on very small corpora), the baseline abstains rather than pick
arbitrary lex-first nodes.

### Oracle baselines

Read ground truth (`central_node` or annotations) and therefore do NOT
implement the `ApproachRunner` protocol. They live in
`src/eval/oracle_baselines.py` and are driven by their own script.

| Name                     | What it predicts                                       | What it tells you                          |
|--------------------------|--------------------------------------------------------|--------------------------------------------|
| `central_plus_neighbors` | `central_node ∪ {its 1-hop neighbours, any relation}`  | Recall ceiling for any "anchor + 1-hop" architecture. |
| `gold_only`              | exactly `required ∪ useful`                            | Sanity check on the evaluator (must give F1 = 1.0). |

### Sizing

All sized baselines default to 5 (≈ median `|gold|` on the current
dataset). Override per-baseline in `configs/config.yaml`:

```yaml
approaches:
  random_subset:
    size: 5
    seed: 42
  top_degree:
    size: 5
  bm25:
    size: 5
```

## Inter-annotator agreement

`src/eval/iaa.py` computes IAA on the **raw** `annotations.csv`, NOT the
consolidated dataset (the consolidator already merges votes and drops
`irrelevant` labels — the multi-annotator signal IAA needs is gone there).

What gets reported:

* **Cohen's κ** — pairwise, chance-corrected on three labels.
* **Fleiss' κ** — multi-rater extension when ≥ 3 annotators saw the same sample.
* **Percent agreement** — pairwise agreement uncorrected for chance.

Universe of nodes per pair: by default the union of node_ids labelled by
either annotator, with implicit `irrelevant` for nodes one annotator
didn't tick (matches the labelling UI's "I saw the candidate and decided
not to include it" semantics). Pass `--policy intersection` for the
strict alternative.

Reading the numbers (Landis & Koch 1977 conventions):

| κ value      | Interpretation        |
|--------------|-----------------------|
| ≤ 0.20       | poor / slight         |
| 0.21 – 0.40  | fair                  |
| 0.41 – 0.60  | moderate              |
| 0.61 – 0.80  | substantial           |
| 0.81 – 1.00  | almost perfect        |

If your mean κ lands around 0.5, no system can outperform that on average
— the upper bound is the task itself, not your code.

## Repository layout

```
uml-pruner/
├── README.md
├── requirements.txt              # core deps (LLM pipeline + rank-bm25)
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
├── reports/                      # COMMITTED snapshots of evaluation runs
│   └── README.md                 # naming convention + workflow
│
├── scripts/                      # ENTRY POINTS — only thin CLI wrappers
│   ├── normalize_diagrams.py     # uml_with_methods/ -> data/diagrams_normalized/
│   ├── build_dataset.py          # annotations.csv  -> data/dataset.csv
│   ├── build_index.py            # build embedding indices
│   ├── retrieve.py               # RAG: query an index (debug)
│   ├── eval_retriever.py         # RAG: recall against the dataset
│   ├── run.py                    # GENERIC: run any approach over the dataset
│   ├── eval.py                   # evaluate an existing results dir
│   ├── ablation.py               # side-by-side comparison across approaches
│   ├── iaa.py                    # inter-annotator agreement on annotations.csv
│   └── run_oracle_baselines.py   # oracle (ground-truth-aware) baselines
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
│   │   ├── evaluator.py          # evaluate a results dir against the dataset
│   │   ├── iaa.py                # Cohen's κ / Fleiss' κ / percent agreement
│   │   └── oracle_baselines.py   # central_plus_neighbors, gold_only (read GT)
│   │
│   └── approaches/
│       ├── __init__.py           # registry of approaches + baselines
│       ├── _common/              # cross-approach helpers
│       │   ├── compressor.py     # filter_subgraph + class representation builders
│       │   ├── package_grouper.py
│       │   └── batching.py
│       │
│       ├── baselines/            # empty / full_diagram / random_subset / top_degree / bm25
│       │   ├── runner.py
│       │   ├── __init__.py
│       │   └── README.md
│       │
│       ├── rag_classes_filter/   # #1 — implemented
│       ├── anchor_neighbors/     # #2 — implemented
│       ├── agentic_chunks/       # #3 — stub
│       └── human_like_agent/     # #4 — stub
│
├── tests/
│   ├── unit/                     # autosplit, budget, compressor, metrics, baselines, IAA, oracles
│   ├── integration/              # mocked pipeline + baselines smoke
│   └── fixtures/tiny/            # tiny synthetic dataset for CI
│
├── .github/workflows/ci.yml      # unit tests + CLI smoke runs (no torch, no LLM)
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
- `approaches.<name>` — per-approach knobs. Examples:
  - `approaches.anchor_neighbors.n_candidates`
  - `approaches.random_subset.size` / `.seed`
  - `approaches.top_degree.size`
  - `approaches.bm25.size`
  Absent sections fall back to hard-coded defaults inside each runner.
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
# List all registered approaches (real + baselines)
python scripts/run.py --list

# Real approaches
python scripts/run.py --approach rag_classes_filter
python scripts/run.py --approach anchor_neighbors

# Baselines (no LLM calls, no embedding model needed)
python scripts/run.py --approach empty
python scripts/run.py --approach full_diagram
python scripts/run.py --approach random_subset
python scripts/run.py --approach top_degree
python scripts/run.py --approach bm25

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

### Run oracle baselines (read ground truth)

```bash
# Both oracles, then evaluate. Output: data/results/oracle_<name>/
python scripts/run_oracle_baselines.py

# Pick one
python scripts/run_oracle_baselines.py --baselines central_plus_neighbors

# Restrict / limit
python scripts/run_oracle_baselines.py --repo apache/hadoop
python scripts/run_oracle_baselines.py --limit 5
```

Each result file is marked `metadata.is_oracle = true` so it's never
mistaken for a real approach result.

### Cross-approach ablation

`scripts/ablation.py` runs N approaches over the same dataset and prints
a side-by-side comparison table (F1, recall, precision, mean output
size) — with optional bootstrap 95% CIs on F1.

```bash
# Compare all baselines + the real approach
python scripts/ablation.py \
    --approaches empty random_subset top_degree bm25 anchor_neighbors

# With bootstrap CIs (recommended for the thesis)
python scripts/ablation.py \
    --approaches empty random_subset bm25 anchor_neighbors \
    --bootstrap 1000

# Pull oracle baselines (already generated) into the same table
python scripts/ablation.py \
    --approaches empty random_subset bm25 anchor_neighbors \
    --skip-generation \
    --include-existing-dir oracle_gold_only=data/results/oracle_gold_only \
    --include-existing-dir oracle_central_plus_neighbors=data/results/oracle_central_plus_neighbors

# Restrict / limit
python scripts/ablation.py --approaches empty bm25 --limit 5
python scripts/ablation.py --approaches empty bm25 --repo apache/hadoop
```

The script does **not** regenerate result files that already exist (so
reruns are cheap). Pass `--overwrite` to force regeneration.

### Inter-annotator agreement

```bash
# Default: read annotations.csv at the repo root, finalized rows only.
python scripts/iaa.py

# Custom path / save full per-sample report
python scripts/iaa.py \
    --annotations annotations.csv \
    --output      reports/$(date +%Y-%m-%d).iaa.json

# Drop a specific annotator (mirror of build_dataset.py)
python scripts/iaa.py --exclude-annotator AndrewRatkov

# Use intersection-of-labelled-nodes instead of union+implicit-irrelevant
python scripts/iaa.py --policy intersection
```

What it prints:

```
============================================================
INTER-ANNOTATOR AGREEMENT
============================================================
  Samples with >= 2 annotators: <N>
  Pairwise comparisons:         <K>
  Universe policy:              union_with_implicit_irrelevant

  Mean Cohen's κ:        0.xxx
  Mean percent agreement:0.xxx
  Mean Fleiss' κ:        0.xxx
============================================================
```

### Snapshot a run to `reports/`

After each meaningful change (prompt tweak, new approach, bigger
annotation pool), save and commit:

```bash
TAG="$(date +%Y-%m-%d)-short-description"

python scripts/ablation.py \
    --approaches empty random_subset top_degree bm25 anchor_neighbors \
    --bootstrap 1000 \
    --output "reports/${TAG}.ablation.json"

python scripts/iaa.py --output "reports/${TAG}.iaa.json"

git add reports/
git commit -m "Snapshot ${TAG}"
```

`reports/` is the canonical place for committed evaluation snapshots. See
`reports/README.md` for the naming convention.

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

For the multi-annotator agreement signal that the consolidator drops
(`irrelevant` votes, per-annotator labels), use `scripts/iaa.py` instead;
it works on the raw `annotations.csv`.

### Ground-truth isolation

`ApproachInputs` deliberately omits `central_node` and the per-sample
annotations. Approaches see ONLY the user query and the full normalized
diagram — same information as in production. `sample_id` and `repo` are
passed in for output filenames and on-disk index lookup, and runners must
not derive any decision from them.

Oracle baselines (`src/eval/oracle_baselines.py`) intentionally violate
this invariant — they exist to give upper bounds and sanity checks. They
do NOT go through `ApproachRunner` and are clearly tagged
`metadata.is_oracle = true` in their output, so they can never be confused
with a real approach result.

## Continuous integration

`.github/workflows/ci.yml` runs on Python 3.10 / 3.11 / 3.12 and exercises:

- unit tests across `src/eval`, `src/approaches/baselines`, and metrics;
- integration smoke (baselines on a synthetic micro-dataset in
  `tests/fixtures/tiny/`);
- the `scripts/iaa.py` CLI on the tiny fixture;
- the `scripts/run_oracle_baselines.py` CLI (verifies F1 = 1.0 for
  `gold_only`);
- the `scripts/ablation.py` CLI on all five baselines.

It deliberately does **not** install `torch` / `sentence-transformers`,
because those (a) take 5–10 minutes per workflow, (b) require multi-GB
model downloads, and (c) aren't needed to verify any of the contracts
above. Local checks for the embedding stack remain the developer's
responsibility before each push.

CI also does not run the existing `tests/integration/test_pipeline_mock.py`:
it defines `async def test_*` (would need `pytest-asyncio`) and loads a
gitignored production diagram. Enable it later with a separate workflow if
needed.

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