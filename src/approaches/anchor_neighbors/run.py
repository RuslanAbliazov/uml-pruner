#!/usr/bin/env python3
"""CLI for running the ``anchor_neighbors`` approach over the dataset.

This is a self-contained entry point for approach #2 — invoke it directly
from this folder instead of going through ``scripts/run.py``.

Usage
-----

    # Whole dataset, then evaluate
    python src/approaches/anchor_neighbors/run.py

    # A handful of samples, no eval
    python src/approaches/anchor_neighbors/run.py --limit 5 --no-eval

    # One repo / one sample
    python src/approaches/anchor_neighbors/run.py --repo apache/hadoop
    python src/approaches/anchor_neighbors/run.py --sample-id <id>

Outputs go to ``data/results/anchor_neighbors/`` by default.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Make the repo root importable when this file is run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from tqdm import tqdm

from src.approaches.anchor_neighbors import build_runner
from src.core.config import load_config
from src.core.io import load_diagram, save_diagram, save_json
from src.core.logger import setup_logger
from src.core.types import ApproachInputs
from src.eval.annotations import diagram_filename_for_repo, load_dataset
from src.eval.evaluator import evaluate_test_set, format_summary_report

APPROACH_NAME = "anchor_neighbors"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Run the '{APPROACH_NAME}' approach.")
    p.add_argument("--dataset", default="data/dataset.csv",
                   help="Consolidated dataset CSV (default: data/dataset.csv).")
    p.add_argument("--diagrams-dir", default="data/diagrams_normalized",
                   help="Where normalized diagrams live "
                        "(default: data/diagrams_normalized).")
    p.add_argument("--output-dir", default=None,
                   help=f"Where to write per-sample results. "
                        f"Default: data/results/{APPROACH_NAME}/.")
    p.add_argument("--config", default="configs/config.yaml",
                   help="Project YAML config.")
    p.add_argument("--repo", default="",
                   help="Restrict to samples from this repo slug "
                        "(e.g. 'apache/hadoop').")
    p.add_argument("--sample-id", default="",
                   help="Run only the sample with this id (debugging).")
    p.add_argument("--limit", type=int, default=0,
                   help="Process at most N samples (0 = all).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run even if a result file already exists.")
    p.add_argument("--no-eval", action="store_true",
                   help="Skip the evaluation step after generation.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


async def _process_sample(
    runner,
    sample,
    diagrams_dir: Path,
    diagram_cache: dict,
    output_dir: Path,
    overwrite: bool,
    errors: list[dict],
) -> bool:
    """Run the runner on a single sample. Returns True iff we wrote a file."""
    out_file = output_dir / f"{sample.sample_id}.json"
    if out_file.exists() and not overwrite:
        return False

    fname = diagram_filename_for_repo(sample.repo)
    if not fname:
        errors.append({"sample_id": sample.sample_id, "reason": "unknown_repo",
                       "repo": sample.repo})
        return False
    path = diagrams_dir / fname
    if not path.exists():
        errors.append({"sample_id": sample.sample_id, "reason": "diagram_missing",
                       "path": str(path)})
        return False
    if fname not in diagram_cache:
        diagram_cache[fname] = load_diagram(path)
    diagram = diagram_cache[fname]

    # NOTE: ground-truth (central_node) is intentionally NOT passed in.
    # The approach sees only the user query + the full diagram.
    inputs = ApproachInputs(
        query=sample.query,
        diagram=diagram,
        sample_id=sample.sample_id,
        repo=sample.repo,
    )

    try:
        result = await runner.run(inputs)
    except Exception as e:  # noqa: BLE001 — surface failures as errors
        errors.append({"sample_id": sample.sample_id,
                       "reason": "runner_error", "detail": repr(e)})
        return False

    diagram_out = result.to_diagram()
    meta = diagram_out.setdefault("metadata", {})
    meta["sample_id"] = sample.sample_id
    meta["repo"] = sample.repo
    meta["query"] = sample.query
    meta["approach"] = APPROACH_NAME
    save_diagram(diagram_out, out_file)
    return True


async def _run(args: argparse.Namespace) -> None:
    load_dotenv()
    cfg = load_config(args.config)
    setup_logger(
        level="DEBUG" if args.verbose else cfg.logging.get("level", "INFO"),
        log_file=cfg.logging.get("file", None),
    )

    samples = load_dataset(args.dataset)
    if args.repo:
        samples = [s for s in samples if s.repo == args.repo]
    if args.sample_id:
        samples = [s for s in samples if s.sample_id == args.sample_id]
    if args.limit:
        samples = samples[: args.limit]

    if not samples:
        print("[error] no samples to run after filtering.", file=sys.stderr)
        sys.exit(2)

    diagrams_dir = Path(args.diagrams_dir)
    output_dir = Path(args.output_dir or f"data/results/{APPROACH_NAME}")
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = build_runner(cfg)

    diagram_cache: dict[str, dict] = {}
    errors: list[dict] = []
    finished = 0

    t0 = time.time()
    for sample in tqdm(samples, desc=APPROACH_NAME):
        if await _process_sample(
            runner, sample, diagrams_dir, diagram_cache,
            output_dir, args.overwrite, errors,
        ):
            finished += 1
    await runner.aclose()

    elapsed = time.time() - t0
    print(f"\nDone: finished={finished}, errors={len(errors)}, "
          f"elapsed={elapsed:.1f}s")
    if errors:
        save_json(errors, output_dir / "errors.json")
        print("  (first 5 errors):")
        for e in errors[:5]:
            print(f"    - {e}")

    if args.no_eval:
        return

    print("\nEvaluating against dataset…")
    eval_result = evaluate_test_set(
        dataset_csv=args.dataset,
        results_dir=str(output_dir),
        result_filename_template="{sample_id}.json",
    )
    print(format_summary_report(eval_result))
    report_path = output_dir / "evaluation_report.json"
    save_json(eval_result.to_dict(), report_path)
    print(f"\nReport: {report_path}")


def main() -> None:
    asyncio.run(_run(parse_args()))


if __name__ == "__main__":
    main()
