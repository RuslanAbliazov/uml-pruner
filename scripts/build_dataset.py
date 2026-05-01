#!/usr/bin/env python3
"""Build the consolidated annotation dataset from raw annotations.csv.

This is the SINGLE place where any data preparation / cleaning / merging of
raw annotations happens. Every other script in the project must consume the
output of this script as immutable, ready-to-use data.

Output schema (CSV, one row per (sample_id, query, central_node)):

    _id, sample_id, task_id, central_node, repo, query, entity_annotations

Where:
    _id                 : a deterministic id (hash of sample_id+central_node).
    sample_id           : sample_id from the raw CSV (or hash fallback).
    task_id             : task_id from the raw CSV.
    central_node        : focus class (node_id) of the sample.
    repo                : repository slug (e.g. "apache/hadoop"); resolved by
                          searching for `central_node` inside each diagram's
                          JSON ("nodes[*].node_id") in --diagrams-dir.
    query               : the natural-language query.
    entity_annotations  : JSON map node_id -> "required" | "useful".
                          "irrelevant" is dropped from the final map.

Voting rules
------------
For each (sample_id) we collect every annotator's `entity_annotations`. For
each node_id mentioned by any annotator, we count votes per label:

    - The winning label is the one with strict majority (> 50% of cast votes
      for that node).
    - If no label has strict majority (e.g. 1/1, 1/1/1), priority is applied:
      required > useful > irrelevant (i.e. if any annotator marked it
      required, the node becomes required; otherwise if any marked it useful,
      it becomes useful; otherwise irrelevant).
    - "irrelevant" labels never appear in the output map.

Filtering
---------
    Only rows with status == "Finalized" are considered. In-progress and
    Not-Annotated rows are dropped unconditionally.

    --min-annotators N         : default 2. Samples with fewer than N distinct
                                 annotators (after exclusion) are dropped,
                                 unless --keep-single is set in which case
                                 single-annotator samples are kept.
    --keep-single              : keep diagrams that ended up with only one
                                 annotation (overrides --min-annotators=2).
    --exclude-annotator NAME   : exclude votes of the given annotator. May be
                                 specified multiple times.
    --include-non-finalized    : (debug) also include rows whose status is not
                                 "Finalized". Off by default.

Usage
-----
    python scripts/build_dataset.py \
        --annotations annotations.csv \
        --diagrams-dir full_diagrams_fixed_generic \
        --output data/dataset.csv

    # Exclude annotators and keep single-annotator samples:
    python scripts/build_dataset.py \
        --exclude-annotator AndrewRatkov \
        --exclude-annotator TRuslan \
        --keep-single
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------------------------------------------------------
# Output / dataset row shape
# ----------------------------------------------------------------------------

DATASET_FIELDS = [
    "_id",
    "sample_id",
    "task_id",
    "central_node",
    "repo",
    "query",
    "entity_annotations",
]


# ----------------------------------------------------------------------------
# Repo lookup: scan diagrams once, build node_id -> repo map
# ----------------------------------------------------------------------------


@dataclass
class DiagramInfo:
    diagram_file: str  # e.g. "hadoop.json"
    project: str  # filename stem, e.g. "hadoop"
    repo: str  # e.g. "apache/hadoop"
    node_ids: set[str]


# Hard-coded mapping diagram-file-stem -> repository slug. Diagrams come from
# real GitHub projects; this is just metadata so we can fill the `repo` column.
# If a diagram is added that is not listed here, we fall back to its stem.
_DIAGRAM_REPO_MAP = {
    "hadoop": "apache/hadoop",
    "flink": "apache/flink",
    "ghidra": "NationalSecurityAgency/ghidra",
    "dbeaver": "dbeaver/dbeaver",
    "deeplearning4j": "deeplearning4j/deeplearning4j",
    "Activiti": "Activiti/Activiti",
    "disruptor": "LMAX-Exchange/disruptor",
    "thingsboard": "thingsboard/thingsboard",
}


def _diagram_to_repo(diagram_stem: str) -> str:
    return _DIAGRAM_REPO_MAP.get(diagram_stem, diagram_stem)


def index_diagrams(diagrams_dir: Path) -> list[DiagramInfo]:
    """Load every JSON diagram in `diagrams_dir` and index its node_ids.

    Returns a list of DiagramInfo. Each DiagramInfo holds the set of
    `node_id`s present in that diagram, used to resolve which repo a
    given `central_node` belongs to.
    """
    if not diagrams_dir.exists():
        raise FileNotFoundError(f"Diagrams directory not found: {diagrams_dir}")

    out: list[DiagramInfo] = []
    for path in sorted(diagrams_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[warn] could not parse {path}: {e}", file=sys.stderr)
            continue

        node_ids: set[str] = set()
        for n in data.get("nodes", []):
            nid = n.get("node_id")
            if nid:
                node_ids.add(nid)
        stem = path.stem
        out.append(
            DiagramInfo(
                diagram_file=path.name,
                project=stem,
                repo=_diagram_to_repo(stem),
                node_ids=node_ids,
            )
        )
    return out


def resolve_repo(central_node: str, diagrams: list[DiagramInfo]) -> tuple[str, str]:
    """Return (repo_slug, diagram_file) for `central_node`.

    The diagram is the one whose `nodes[*].node_id` set contains the central
    node. If multiple match (shouldn't happen if FQNs are unique across
    projects), the first match wins. If none matches, ("", "") is returned.
    """
    for d in diagrams:
        if central_node in d.node_ids:
            return d.repo, d.diagram_file
    return "", ""


# ----------------------------------------------------------------------------
# Raw CSV parsing
# ----------------------------------------------------------------------------


@dataclass
class RawRow:
    sample_id: str
    task_id: str
    annotator: str
    central_node: str
    query: str
    annotations: dict[str, str]
    status: str


def _parse_annotations_field(raw: str) -> dict[str, str]:
    if not raw or raw == "{}":
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback for double-escaped quotes occasionally seen in raw CSVs.
        return json.loads(raw.replace('""', '"'))


def _hash_id(*parts: str) -> str:
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return h[:24]


def load_raw_rows(csv_path: Path) -> list[RawRow]:
    rows: list[RawRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ann = _parse_annotations_field(r.get("entity_annotations", ""))
            sid = r.get("sample_id", "") or r.get("_id", "")
            cn = r.get("central_node", "") or ""
            q = r.get("query", "") or ""
            if not sid:
                # Fallback: derive a stable sample_id from (central_node, query).
                sid = _hash_id(cn, q)
            rows.append(
                RawRow(
                    sample_id=sid,
                    task_id=r.get("task_id", "") or "",
                    annotator=r.get("annotator", "") or "",
                    central_node=cn,
                    query=q,
                    annotations=ann,
                    status=r.get("status", "") or "",
                )
            )
    return rows


# ----------------------------------------------------------------------------
# Voting
# ----------------------------------------------------------------------------

_VALID_LABELS = ("required", "useful", "irrelevant")
_PRIORITY = {"required": 3, "useful": 2, "irrelevant": 1}


def vote_annotations(per_annotator: dict[str, dict[str, str]]) -> dict[str, str]:
    """Aggregate per-annotator label dicts into a single voted dict.

    Args:
        per_annotator: {annotator_name -> {node_id -> label}}.

    Rules:
        - Each annotator casts at most one vote per node_id (the label they
          assigned). Annotators who didn't mention a node don't vote on it.
        - The winning label is the one with strict majority (> 50%) of the
          cast votes for that node.
        - If no label has strict majority, fall back to label priority:
          required > useful > irrelevant. (Any required vote wins, else
          any useful vote wins, else irrelevant.)
        - Output drops "irrelevant" entries.

    Returns:
        {node_id -> "required" | "useful"} (no irrelevant).
    """
    # Collect votes per node_id
    votes_by_node: dict[str, list[str]] = defaultdict(list)
    for _annotator, labels in per_annotator.items():
        for nid, lbl in labels.items():
            if lbl not in _VALID_LABELS:
                continue
            votes_by_node[nid].append(lbl)

    voted: dict[str, str] = {}
    for nid, votes in votes_by_node.items():
        total = len(votes)
        if total == 0:
            continue
        counts = Counter(votes)
        # strict majority
        winner = None
        for lbl, c in counts.items():
            if c * 2 > total:  # c > total/2  (strict)
                winner = lbl
                break
        if winner is None:
            # tie / no majority -> priority required > useful > irrelevant
            for lbl in ("required", "useful", "irrelevant"):
                if counts.get(lbl, 0) > 0:
                    winner = lbl
                    break
        if winner is None:
            continue
        if winner == "irrelevant":
            continue  # dropped from final map
        voted[nid] = winner
    return voted


# ----------------------------------------------------------------------------
# Build dataset
# ----------------------------------------------------------------------------


@dataclass
class BuildStats:
    raw_rows: int = 0
    rows_after_status_filter: int = 0
    rows_after_exclusion: int = 0
    samples_total: int = 0
    samples_kept: int = 0
    samples_dropped_no_annotations: int = 0
    samples_dropped_too_few_annotators: int = 0
    samples_dropped_unknown_repo: int = 0


def build_dataset(
    raw_rows: list[RawRow],
    diagrams: list[DiagramInfo],
    excluded_annotators: set[str],
    min_annotators: int,
    keep_single: bool,
    finalized_only: bool = True,
) -> tuple[list[dict[str, Any]], BuildStats]:
    stats = BuildStats(raw_rows=len(raw_rows))

    # 1. keep only Finalized rows (the only ones whose annotations are trusted).
    if finalized_only:
        rows = [r for r in raw_rows if r.status.lower() == "finalized"]
    else:
        rows = list(raw_rows)
    stats.rows_after_status_filter = len(rows)

    # 2. drop excluded annotators' rows.
    rows = [r for r in rows if r.annotator not in excluded_annotators]
    stats.rows_after_exclusion = len(rows)

    # 3. group by sample_id.
    grouped: dict[str, list[RawRow]] = defaultdict(list)
    for r in rows:
        grouped[r.sample_id].append(r)
    stats.samples_total = len(grouped)

    out: list[dict[str, Any]] = []
    for sample_id, sample_rows in grouped.items():
        # Annotators who actually produced labels for this sample.
        labelled_rows = [r for r in sample_rows if r.annotations]
        # An empty `annotator` field means "raw template row" (status often
        # 'Not Annotated'); we ignore those.
        labelled_rows = [r for r in labelled_rows if r.annotator]

        if not labelled_rows:
            stats.samples_dropped_no_annotations += 1
            continue

        # Distinct annotators (a single annotator may have multiple rows for
        # the same sample, e.g. progressive saves). For voting, take their
        # most-recent (= last) annotation snapshot.
        per_annotator: dict[str, dict[str, str]] = {}
        for r in labelled_rows:
            per_annotator[r.annotator] = r.annotations

        n_annotators = len(per_annotator)
        if n_annotators < min_annotators and not (keep_single and n_annotators >= 1):
            stats.samples_dropped_too_few_annotators += 1
            continue

        first = labelled_rows[0]
        repo, _diagram_file = resolve_repo(first.central_node, diagrams)
        if not repo:
            stats.samples_dropped_unknown_repo += 1
            print(
                f"[warn] central_node not found in any diagram: "
                f"sample_id={sample_id} central_node={first.central_node}",
                file=sys.stderr,
            )
            continue

        voted = vote_annotations(per_annotator)
        record_id = _hash_id(sample_id, first.central_node)

        out.append(
            {
                "_id": record_id,
                "sample_id": sample_id,
                "task_id": first.task_id,
                "central_node": first.central_node,
                "repo": repo,
                "query": first.query,
                "entity_annotations": json.dumps(voted, ensure_ascii=False),
            }
        )
        stats.samples_kept += 1

    out.sort(key=lambda r: (r["repo"], r["sample_id"]))
    return out, stats


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the consolidated annotation dataset (single source of truth)."
    )
    p.add_argument(
        "--annotations",
        default="annotations.csv",
        help="Path to the raw annotations CSV (default: annotations.csv).",
    )
    p.add_argument(
        "--diagrams-dir",
        default="full_diagrams_fixed_generic",
        help="Directory with project diagram JSONs (default: full_diagrams_fixed_generic).",
    )
    p.add_argument(
        "--output",
        default="data/dataset.csv",
        help="Output CSV path (default: data/dataset.csv).",
    )
    p.add_argument(
        "--min-annotators",
        type=int,
        default=2,
        help="Minimum number of distinct annotators per sample (default: 2).",
    )
    p.add_argument(
        "--keep-single",
        action="store_true",
        help="Keep samples with only a single annotator (overrides --min-annotators=2).",
    )
    p.add_argument(
        "--exclude-annotator",
        action="append",
        default=[],
        help="Exclude votes of this annotator. May be passed multiple times.",
    )
    p.add_argument(
        "--include-non-finalized",
        action="store_true",
        help="(Debug) Also include rows whose status is not 'Finalized'. "
        "By default only Finalized rows are used.",
    )
    p.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a build-stats summary at the end.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    annotations_path = Path(args.annotations)
    diagrams_dir = Path(args.diagrams_dir)
    output_path = Path(args.output)

    if not annotations_path.exists():
        print(f"[error] annotations CSV not found: {annotations_path}", file=sys.stderr)
        sys.exit(2)

    diagrams = index_diagrams(diagrams_dir)
    if not diagrams:
        print(f"[error] no diagrams found in {diagrams_dir}", file=sys.stderr)
        sys.exit(2)

    raw_rows = load_raw_rows(annotations_path)
    excluded = set(args.exclude_annotator or [])

    rows, stats = build_dataset(
        raw_rows=raw_rows,
        diagrams=diagrams,
        excluded_annotators=excluded,
        min_annotators=args.min_annotators,
        keep_single=args.keep_single,
        finalized_only=not args.include_non_finalized,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} sample(s) to {output_path}")
    if args.print_summary or excluded or args.keep_single or args.include_non_finalized:
        print()
        print("Build summary:")
        print(f"  raw rows:                       {stats.raw_rows}")
        print(f"  rows after status filter:       {stats.rows_after_status_filter}")
        print(f"  rows after annotator exclusion: {stats.rows_after_exclusion}")
        print(f"  samples (unique sample_id):     {stats.samples_total}")
        print(f"  samples kept:                   {stats.samples_kept}")
        print(f"  dropped (no annotations):       {stats.samples_dropped_no_annotations}")
        print(
            f"  dropped (too few annotators):   "
            f"{stats.samples_dropped_too_few_annotators}"
        )
        print(f"  dropped (unknown repo):         {stats.samples_dropped_unknown_repo}")
        if excluded:
            print(f"  excluded annotators:            {sorted(excluded)}")
        print(
            f"  rule: finalized_only={not args.include_non_finalized}, "
            f"keep_single={args.keep_single}, min_annotators={args.min_annotators}"
        )


if __name__ == "__main__":
    main()
