# Snapshots of evaluation results

This directory holds time-stamped JSON snapshots of evaluation runs. The
goal is to have a git-tracked history of metrics for the thesis defense
rather than "trust me, I ran it last week" — and to make sure the
advisor / committee can see fresh numbers without ssh-ing into your
machine.

## Why here and not `data/results/`

`data/` is gitignored on purpose — it holds the raw dataset and per-sample
prediction JSONs, which are large and rebuilt on every run. `reports/`
holds only the small aggregated JSONs you want preserved.

## Convention

After every meaningful change (prompt tweak, new approach, bigger
annotation pool), save and commit:

```bash
TAG="$(date +%Y-%m-%d)-short-description"
# e.g. TAG=2026-05-12-after-stage4-prompt-fix

# Ablation across baselines + your real approaches:
python scripts/ablation.py \
    --approaches empty random_subset top_degree bm25 anchor_neighbors \
    --bootstrap 1000 \
    --output "reports/${TAG}.ablation.json"

# IAA on the latest annotations:
python scripts/iaa.py --output "reports/${TAG}.iaa.json"

git add "reports/${TAG}.ablation.json" "reports/${TAG}.iaa.json"
git commit -m "Snapshot ${TAG}"
```

Filename pattern: `<YYYY-MM-DD>-<short-description>.<kind>.json` —
date first so `ls reports/` sorts chronologically.

## What kinds of snapshot live here

| Kind         | Source script                       | Size       |
|--------------|--------------------------------------|------------|
| `*.ablation.json` | `scripts/ablation.py --output ...` | ~1–3 KB    |
| `*.iaa.json`      | `scripts/iaa.py --output ...`      | ~1–10 KB   |

Both are pure JSON — easy to diff in git, easy to load into a notebook for
plots, easy to read by hand.

## Reading the diffs

`git diff` over two snapshot files shows exactly what changed:

```bash
git diff reports/2026-05-10-baseline.ablation.json reports/2026-05-12-after-prompt-fix.ablation.json
```

Recommended for the thesis: a small table tracking F1 of `anchor_neighbors`
(and key baselines) over time, sourced from these snapshots. Makes it
trivial to defend "we improved by X across change Y" with a paper trail.

## What NOT to put here

* Per-sample prediction JSONs (`data/results/<approach>/<sample_id>.json`)
  — too many, too large, regenerable.
* Notebooks, plots, PDFs — those go in a separate `paper/` or `notebooks/`
  directory if you want them.
* The raw dataset (`data/dataset.csv`) or diagrams — these are gitignored
  and live in your local `data/` only.
