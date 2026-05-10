# Baselines

A set of dirt-cheap reference approaches to anchor every "real" approach's
metrics. None of them call an LLM or load an embedding model, so they run
on a laptop in seconds and in CI without any extra setup.

Two flavours:

| Flavour       | Sees ground truth? | Goes through `ApproachRunner`? | Where to find it |
|---------------|---|---|---|
| Query-agnostic| No  | Yes — registered in `REGISTRY` | this package |
| Oracle        | Yes | No  — direct script         | `src/eval/oracle_baselines.py` |

The oracle baselines violate the project's "approach must not see central_node
or annotations" invariant by design. They tell you what F1 you'd get if you
started from privileged information, which is useful as an upper bound but
must never be reported as a real approach result.

## Query-agnostic baselines

| Name           | What it predicts                                  |
|----------------|---------------------------------------------------|
| `empty`        | nothing                                           |
| `full_diagram` | the entire diagram                                |
| `random_subset`| `size` node_ids picked uniformly at random        |
| `top_degree`   | top-`size` nodes by total degree (in+out)         |

## Lexical baseline

| Name   | What it predicts                                                                |
|--------|---------------------------------------------------------------------------------|
| `bm25` | top-`size` nodes by BM25 against the query, over the same node-text serialization (`src.rag.node_to_text.nodes_to_texts`) the embedding retriever uses. |

`bm25` is the classical sparse retrieval comparison for any dense retriever
work: if the dense retriever doesn't beat BM25 on this dataset, the
embedding stack is not earning its keep. The tokenizer splits CamelCase /
snake_case to lowercase pieces and drops single-char tokens, so
`HashMapImpl` matches a query of `hash map`. Powered by the small
`rank_bm25` package (added to `requirements.txt`).

Configurable via `configs/config.yaml` (all keys optional):

```yaml
approaches:
  random_subset:
    size: 5    # default ≈ median(|gold|) on the dataset
    seed: 42
  top_degree:
    size: 5
  bm25:
    size: 5
```

`random_subset` is reproducibly seeded per `sample_id` (via
`hashlib.blake2b`), so re-running the same dataset gives bit-exact output.

### Run them

```bash
# One baseline end-to-end (generation + evaluation):
python scripts/run.py --approach empty
python scripts/run.py --approach full_diagram
python scripts/run.py --approach random_subset
python scripts/run.py --approach top_degree
python scripts/run.py --approach bm25

# All five side-by-side, plus your real approaches, in one table:
python scripts/ablation.py \
    --approaches empty full_diagram random_subset top_degree bm25 \
                 anchor_neighbors rag_classes_filter \
    --output reports/$(date +%Y-%m-%d).ablation.json
```

`ablation.py` writes a side-by-side comparison table to stdout and a
JSON report you can paste into the thesis.

## Oracle baselines

These need both the diagram and the annotation row, so they don't go
through `scripts/run.py`. Use:

```bash
python scripts/run_oracle_baselines.py \
    --baselines central_plus_neighbors gold_only \
    --dataset      data/dataset.csv \
    --diagrams-dir data/diagrams_normalized \
    --output-root  data/results
```

| Name                     | What it predicts                                    | Why                                |
|--------------------------|-----------------------------------------------------|------------------------------------|
| `central_plus_neighbors` | `central_node ∪ direct neighbours` (1 hop, any rel) | Is "1 hop from the right anchor" enough? |
| `gold_only`              | exactly `required ∪ useful`                         | Sanity: confirms F1 = 1.0 path     |

Reading the oracle numbers:

* If `central_plus_neighbors` already reaches F1 = 0.55, the recall ceiling
  for any "anchor + 1-hop" architecture is around there. To go higher you
  need 2-hop expansion or a different stage-2 strategy.
* If `central_plus_neighbors` only reaches F1 = 0.30, the gold answer
  doesn't sit inside the 1-hop neighbourhood — even of the *right* anchor.
  In that case improving the anchor pick won't save you and you need a
  different graph-traversal strategy entirely.

`gold_only` should always print F1 = 1.0; if it doesn't, your evaluator is
broken (or the output JSON shape diverged from what the evaluator reads).
