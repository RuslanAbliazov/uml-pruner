#!/usr/bin/env python3
"""Проверка покрытия central_node якорными классами.

Для каждой записи в dataset:
  - читает central_node
  - открывает соответствующий файл anchors из stage2_anchors_hashed_1
  - проверяет, входит ли central_node в anchors

Печатает per-sample таблицу и сводные метрики.

Использование:
    python scripts/check_central_in_anchors.py
    python scripts/check_central_in_anchors.py --dataset data/dataset_1_iter_subtract.csv
    python scripts/check_central_in_anchors.py --anchors-dir data/stage2_anchors_hashed_1
    python scripts/check_central_in_anchors.py --repo NationalSecurityAgency/ghidra
    python scripts/check_central_in_anchors.py --show-misses-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "dataset_1_iter_subtract.csv"
DEFAULT_ANCHORS_DIR = PROJECT_ROOT / "data" / "stage2_anchors_hashed_1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Проверить, попадает ли central_node в список якорей",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Путь к dataset CSV",
    )
    p.add_argument(
        "--anchors-dir",
        type=Path,
        default=DEFAULT_ANCHORS_DIR,
        help="Директория с anchor-файлами {repo}__{sample_id}.json",
    )
    p.add_argument(
        "--repo",
        default="",
        help="Фильтр по репозиторию (например, apache/hadoop)",
    )
    p.add_argument(
        "--show-misses-only",
        action="store_true",
        help="Показать только сэмплы, где central_node НЕ попал в якоря",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="Сохранить отчёт в JSON по этому пути (опционально)",
    )
    return p.parse_args()


def load_dataset(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"ERROR: Dataset не найден: {path}", file=sys.stderr)
        sys.exit(1)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_anchors(anchors_dir: Path, repo: str, sample_id: str) -> list[str] | None:
    """Загрузить якоря для сэмпла. Возвращает None если файла нет."""
    fname = f"{repo.replace('/', '_')}__{sample_id}.json"
    fpath = anchors_dir / fname
    if not fpath.exists():
        return None
    with fpath.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("anchors", [])


# ---------- pretty printing helpers ----------

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def color(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{RESET}"


def short(node_id: str, width: int = 60) -> str:
    if len(node_id) <= width:
        return node_id
    return "…" + node_id[-(width - 1):]


def fmt_pct(numer: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{100.0 * numer / denom:5.1f}%"


def bar(numer: int, denom: int, width: int = 30) -> str:
    if denom == 0:
        return "[" + " " * width + "]"
    filled = int(round(width * numer / denom))
    return "[" + "█" * filled + "·" * (width - filled) + "]"


def main() -> int:
    args = parse_args()

    dataset = load_dataset(args.dataset)
    if args.repo:
        dataset = [r for r in dataset if r.get("repo") == args.repo]

    if not dataset:
        print("ERROR: датасет пуст после фильтрации", file=sys.stderr)
        return 1

    per_sample: list[dict] = []
    missing_anchor_files: list[str] = []

    for row in dataset:
        sample_id = row["sample_id"]
        repo = row["repo"]
        central = row["central_node"]

        anchors = load_anchors(args.anchors_dir, repo, sample_id)
        if anchors is None:
            missing_anchor_files.append(f"{repo}__{sample_id}")
            continue

        hit = central in anchors
        try:
            rank = anchors.index(central) + 1 if hit else None
        except ValueError:
            rank = None

        per_sample.append(
            {
                "sample_id": sample_id,
                "repo": repo,
                "central_node": central,
                "n_anchors": len(anchors),
                "hit": hit,
                "rank": rank,
            }
        )

    # ---------- print ----------

    print()
    print(color("=" * 100, BOLD))
    print(color("  CENTRAL NODE COVERAGE BY ANCHORS", BOLD))
    print(color("=" * 100, BOLD))
    print(f"  Dataset:     {args.dataset}")
    print(f"  Anchors dir: {args.anchors_dir}")
    if args.repo:
        print(f"  Repo filter: {args.repo}")
    print(f"  Samples evaluated: {len(per_sample)}")
    if missing_anchor_files:
        print(
            color(
                f"  Missing anchor files: {len(missing_anchor_files)}",
                YELLOW,
            )
        )
    print()

    # Per-sample table
    rows_to_show = per_sample
    if args.show_misses_only:
        rows_to_show = [r for r in per_sample if not r["hit"]]

    if rows_to_show:
        print(color("-" * 100, DIM))
        header = f"  {'#':>3}  {'HIT':<5}  {'RANK':<5}  {'REPO':<28}  {'SAMPLE_ID':<26}  CENTRAL_NODE"
        print(color(header, BOLD))
        print(color("-" * 100, DIM))

        for i, r in enumerate(rows_to_show, 1):
            mark = color(" ✓  ", GREEN) if r["hit"] else color(" ✗  ", RED)
            rank_str = (
                color(f"{r['rank']:>3}", GREEN) if r["hit"] else color("  -", RED)
            )
            repo = r["repo"][:28]
            sid = r["sample_id"][:26]
            central_disp = short(r["central_node"], 35)
            print(
                f"  {i:>3}  {mark}  {rank_str}    {repo:<28}  {sid:<26}  {central_disp}"
            )
        print(color("-" * 100, DIM))

    # ---------- summary ----------

    total = len(per_sample)
    hits = sum(1 for r in per_sample if r["hit"])
    misses = total - hits

    print()
    print(color("=" * 100, BOLD))
    print(color("  SUMMARY", BOLD))
    print(color("=" * 100, BOLD))
    print(f"  Total samples:    {total}")
    print(
        f"  Central in anchors: {color(str(hits), GREEN)}  "
        f"{bar(hits, total)}  {fmt_pct(hits, total)}"
    )
    print(
        f"  Central missed:     {color(str(misses), RED)}  "
        f"{bar(misses, total)}  {fmt_pct(misses, total)}"
    )

    # Rank distribution among hits
    if hits > 0:
        ranks = [r["rank"] for r in per_sample if r["hit"]]
        ranks.sort()
        avg_rank = sum(ranks) / len(ranks)
        median_rank = ranks[len(ranks) // 2]
        print()
        print(color("  Rank of central_node among anchors (when hit):", BOLD))
        print(f"    min:    {min(ranks)}")
        print(f"    median: {median_rank}")
        print(f"    avg:    {avg_rank:.2f}")
        print(f"    max:    {max(ranks)}")

        # Bucketed distribution
        buckets = {"1": 0, "2-3": 0, "4-5": 0, "6-10": 0, "11+": 0}
        for rk in ranks:
            if rk == 1:
                buckets["1"] += 1
            elif rk <= 3:
                buckets["2-3"] += 1
            elif rk <= 5:
                buckets["4-5"] += 1
            elif rk <= 10:
                buckets["6-10"] += 1
            else:
                buckets["11+"] += 1
        print()
        print(color("  Rank bucket distribution:", BOLD))
        for k, v in buckets.items():
            print(f"    rank {k:<5}  {v:>3}  {bar(v, hits, width=24)}  {fmt_pct(v, hits)}")

    # Per-repo breakdown
    repos: dict[str, list[dict]] = {}
    for r in per_sample:
        repos.setdefault(r["repo"], []).append(r)

    if len(repos) > 1:
        print()
        print(color("  Per-repo coverage:", BOLD))
        max_repo = max(len(r) for r in repos)
        for repo_name, items in sorted(repos.items()):
            h = sum(1 for it in items if it["hit"])
            t = len(items)
            print(
                f"    {repo_name:<{max_repo}}  "
                f"{h}/{t}  {bar(h, t, width=20)}  {fmt_pct(h, t)}"
            )

    if missing_anchor_files:
        print()
        print(color(f"  Missing anchor files ({len(missing_anchor_files)}):", YELLOW))
        for name in missing_anchor_files[:10]:
            print(f"    - {name}")
        if len(missing_anchor_files) > 10:
            print(f"    ... +{len(missing_anchor_files) - 10} more")

    print(color("=" * 100, BOLD))
    print()

    # ---------- optional JSON output ----------

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "dataset": str(args.dataset),
            "anchors_dir": str(args.anchors_dir),
            "summary": {
                "total": total,
                "hits": hits,
                "misses": misses,
                "hit_rate": hits / total if total else 0.0,
            },
            "per_sample": per_sample,
            "missing_anchor_files": missing_anchor_files,
        }
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  JSON report saved: {args.output}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
