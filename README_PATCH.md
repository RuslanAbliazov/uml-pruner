# Baselines + IAA + ablation patch

Adds three pieces to the project:

1. **Query-agnostic baselines** (`empty`, `full_diagram`, `random_subset`,
   `top_degree`) registered next to the real approaches.
2. **Oracle baselines** (`central_plus_neighbors`, `gold_only`) — read
   ground truth, used as upper bounds and sanity checks. Live under
   `src/eval/` (NOT `src/approaches/`) because they intentionally violate
   the no-leakage invariant.
3. **Inter-annotator agreement** — pairwise Cohen's κ, Fleiss' κ, percent
   agreement on the raw `annotations.csv`.

Plus an **ablation script** that runs N approaches and prints a
side-by-side comparison table (with optional bootstrap 95% CI on F1), and
**GitHub Actions CI** that runs unit tests + smoke runs of all three CLIs
on a synthetic micro-dataset (no torch, no LLM).

---

## What's in the patch

```
NEW:
  src/approaches/baselines/__init__.py
  src/approaches/baselines/runner.py
  src/approaches/baselines/README.md
  src/eval/oracle_baselines.py
  src/eval/iaa.py
  scripts/iaa.py
  scripts/run_oracle_baselines.py
  scripts/ablation.py
  tests/unit/test_baselines.py
  tests/unit/test_iaa.py
  tests/unit/test_oracle_baselines.py
  tests/integration/test_baselines_smoke.py
  tests/fixtures/tiny/dataset.csv
  tests/fixtures/tiny/annotations_raw.csv
  tests/fixtures/tiny/diagrams_normalized/tiny.json
  .github/workflows/ci.yml

MODIFIED (one tiny diff, see src_approaches___init__.patch):
  src/approaches/__init__.py    # register the four baselines
```

## Applying the patch

```bash
# 1. Drop in the new files. From the repo root:
unzip -o uml-pruner-baselines-iaa-patch.zip -d .

# 2. Apply the one-liner change to src/approaches/__init__.py:
patch -p1 < src_approaches___init__.patch

# 3. Verify (everything else in this README assumes this passes):
pytest tests/unit -v
```

If `patch` complains about already-applied hunks, the file may have moved
since this patch was authored — re-read `src_approaches___init__.patch`,
it's < 30 lines and trivial to apply by hand.

## What you can run after

### Baselines

```bash
# One baseline end-to-end on the real dataset:
python scripts/run.py --approach empty
python scripts/run.py --approach full_diagram
python scripts/run.py --approach random_subset      # size from configs/config.yaml, default 5
python scripts/run.py --approach top_degree

# Oracle baselines (read ground truth, separate driver):
python scripts/run_oracle_baselines.py
# or pick specific ones:
python scripts/run_oracle_baselines.py --baselines central_plus_neighbors
```

To configure `random_subset` and `top_degree` size, add to `configs/config.yaml`:

```yaml
approaches:
  random_subset:
    size: 5     # ≈ median(|gold|) on the current dataset
    seed: 42
  top_degree:
    size: 5
```

### IAA

```bash
# On the real annotations:
python scripts/iaa.py

# Custom path / save full report:
python scripts/iaa.py \
    --annotations annotations.csv \
    --output data/results/iaa.json

# Drop a specific annotator (mirror of build_dataset.py):
python scripts/iaa.py --exclude-annotator AndrewRatkov
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

### Ablation

```bash
# Side-by-side comparison of baselines vs your real approach:
python scripts/ablation.py \
    --approaches empty full_diagram random_subset top_degree anchor_neighbors

# With 95% CI on F1 (recommended for the thesis):
python scripts/ablation.py \
    --approaches empty random_subset anchor_neighbors \
    --bootstrap 1000

# Pull oracle baseline dirs (already generated) into the same table:
python scripts/ablation.py \
    --approaches empty random_subset anchor_neighbors \
    --skip-generation \
    --include-existing-dir oracle_gold_only=data/results/oracle_gold_only \
    --include-existing-dir oracle_central_plus_neighbors=data/results/oracle_central_plus_neighbors

# Save the full report:
python scripts/ablation.py \
    --approaches empty anchor_neighbors \
    --output data/results/ablation.json
```

Output (a single readable table):

```
=================================================================================================================================
approach             |      n |       F1 |           F1 CI95% |   rec_req |   rec_use |   rec_all |   p_known |  p_strict |  size
---------------------------------------------------------------------------------------------------------------------------------
empty                |     28 |   0.0000 |    [0.000, 0.000]  |    0.0000 |    0.0000 |    0.0000 |    0.0000 |    0.0000 |   0.0
full_diagram         |     28 |   0.0xxx |    [...]           |    1.0000 |    1.0000 |    1.0000 |    1.0000 |    0.0xxx |  ~∞
random_subset        |     28 |   0.0xxx |    [...]           |    ...    |    ...    |    ...    |    ...    |    ...    |   5.0
top_degree           |     28 |   0.0xxx |    [...]           |    ...    |    ...    |    ...    |    ...    |    ...    |   5.0
anchor_neighbors     |     28 |   0.350  |    [0.265, 0.430]  |    0.50   |    0.40   |    0.466  |    0.28   |    ...    |   ~20
=================================================================================================================================
```

The story you want to tell in the thesis is exactly this table. If
`anchor_neighbors` lands clearly above all four query-agnostic baselines
AND inside-or-above the `oracle_central_plus_neighbors` ceiling, your
contribution is well-defined. If it lands at or below one of them, that's
useful information too — and the table shows the reader exactly which.

---

## CI

`.github/workflows/ci.yml` runs on Python 3.10 / 3.11 / 3.12 and exercises:

* unit tests (43 cases),
* integration smoke (6 cases on the tiny fixture),
* `scripts/iaa.py` smoke,
* `scripts/run_oracle_baselines.py` smoke (verifies F1 = 1.0 for `gold_only`),
* `scripts/ablation.py` smoke (verifies all four baseline names land in
  the table).

It deliberately does **not** install torch / sentence-transformers,
because those:
(a) take 5–10 minutes per workflow,
(b) require multi-gigabyte model downloads, and
(c) aren't needed to verify the patch's contracts.

Local checks for the embedding stack remain the student's responsibility
before each push.

## Files in this archive

```
patch/
├── README_PATCH.md                       # this file
├── src_approaches___init__.patch         # apply with `patch -p1 < ...`
├── src/approaches/baselines/...
├── src/eval/iaa.py
├── src/eval/oracle_baselines.py
├── scripts/iaa.py
├── scripts/run_oracle_baselines.py
├── scripts/ablation.py
├── tests/unit/test_baselines.py
├── tests/unit/test_iaa.py
├── tests/unit/test_oracle_baselines.py
├── tests/integration/test_baselines_smoke.py
├── tests/fixtures/tiny/...
└── .github/workflows/ci.yml
```
