#!/usr/bin/env python3
"""Run any registered approach over the dataset and score the results.

Reads:
    - the consolidated dataset (``data/dataset.csv`` by default, built by
      ``scripts/build_dataset.py``);
    - the normalized diagrams (``data/diagrams_normalized/``, built by
      ``scripts/normalize_diagrams.py``).

Writes per-sample result JSONs under ``data/results/<approach>/<sample_id>.json``
and an aggregate evaluation report under
``data/results/<approach>/evaluation_report.json``.

Example:

    # List approaches
    python scripts/run.py --list

    # Run the baseline approach on the whole dataset, then evaluate
    python scripts/run.py --approach rag_classes_filter

    # Limit to N samples or one repo
    python scripts/run.py --approach rag_classes_filter --limit 5
    python scripts/run.py --approach rag_classes_filter --repo apache/hadoop

    # Skip evaluation step
    python scripts/run.py --approach rag_classes_filter --no-eval
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from tqdm import tqdm

from src.approaches import get_runner, list_approaches
from src.core.types import ApproachInputs
from src.eval.annotations import diagram_filename_for_repo, load_dataset
from src.eval.evaluator import evaluate_test_set, format_summary_report
from src.core.config import load_config
from src.core.io import load_diagram, save_diagram, save_json
from src.core.logger import setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark UML-pruning approaches.")
    p.add_argument(
        "--approach",
        help=f"Approach name. One of: {', '.join(list_approaches())}",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List registered approaches and exit.",
    )
    p.add_argument(
        "--dataset",
        default="data/dataset.csv",
        help="Consolidated dataset CSV (default: data/dataset.csv).",
    )
    p.add_argument(
        "--diagrams-dir",
        default="data/diagrams_normalized",
        help="Directory with normalized diagrams (default: data/diagrams_normalized).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to write per-sample results. "
        "Default: data/results/<approach>/.",
    )
    p.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Project YAML config (passed to the approach factory).",
    )
    p.add_argument(
        "--repo",
        default="",
        help="Restrict to samples from this repo slug (e.g. 'apache/hadoop').",
    )
    p.add_argument(
        "--sample-id",
        default="",
        help="Run only the sample with this id (debugging).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N samples (0 = all).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run even if a result file already exists.",
    )
    p.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip the evaluation step after generation.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


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
    output_dir = Path(args.output_dir or f"data/results/{args.approach}")
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = get_runner(args.approach, cfg)

    diagram_cache: dict[str, dict] = {}
    started = 0
    finished = 0
    errors: list[dict] = []

    t0 = time.time()
    for sample in tqdm(samples, desc=f"approach={args.approach}"):
        out_file = output_dir / f"{sample.sample_id}.json"
        if out_file.exists() and not args.overwrite:
            continue

        fname = diagram_filename_for_repo(sample.repo)
        if not fname:
            errors.append(
                {"sample_id": sample.sample_id, "reason": "unknown_repo", "repo": sample.repo}
            )
            continue
        path = diagrams_dir / fname
        if not path.exists():
            errors.append(
                {"sample_id": sample.sample_id, "reason": "diagram_missing", "path": str(path)}
            )
            continue
        if fname not in diagram_cache:
            diagram_cache[fname] = load_diagram(path)
        diagram = diagram_cache[fname]

        # NOTE: we deliberately do NOT pass ``sample.central_node`` to the
        # runner. In production the algorithm only sees the user's query and
        # the full diagram; the focus class is ground-truth used downstream
        # by the evaluator only.
        inputs = ApproachInputs(
            query=sample.query,
            diagram=diagram,
            sample_id=sample.sample_id,
            repo=sample.repo,
        )
        started += 1

        try:
            result = await runner.run(inputs)
        except NotImplementedError as e:
            errors.append(
                {"sample_id": sample.sample_id, "reason": "not_implemented", "detail": str(e)}
            )
            continue
        except Exception as e:  # noqa: BLE001 — surface any other failures as errors
            errors.append(
                {"sample_id": sample.sample_id, "reason": "runner_error", "detail": repr(e)}
            )
            continue

        diagram_out = result.to_diagram()
        # Standard tracking metadata on every result JSON.
        meta = diagram_out.setdefault("metadata", {})
        meta["sample_id"] = sample.sample_id
        meta["repo"] = sample.repo
        meta["query"] = sample.query
        meta["approach"] = args.approach
        save_diagram(diagram_out, out_file)
        finished += 1

    await runner.aclose()
    elapsed = time.time() - t0
    print(
        f"\nDone: started={started}, finished={finished}, errors={len(errors)}, "
        f"elapsed={elapsed:.1f}s"
    )
    if errors:
        # Save an error report; show the first few inline.
        save_json(errors, output_dir / "errors.json")
        print(f"  (first 5 errors):")
        for e in errors[:5]:
            print(f"    - {e}")

    if args.no_eval:
        return

    # ---- evaluate ----
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
    args = parse_args()
    if args.list:
        for n in list_approaches():
            print(n)
        return
    if not args.approach:
        print("[error] --approach is required (or use --list).", file=sys.stderr)
        sys.exit(2)
    if args.approach not in list_approaches():
        print(
            f"[error] unknown approach '{args.approach}'. "
            f"Available: {', '.join(list_approaches())}",
            file=sys.stderr,
        )
        sys.exit(2)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
