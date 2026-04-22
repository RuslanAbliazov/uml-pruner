"""Load and normalize annotations.csv into structured samples."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AnnotationSample:
    """One annotated ground-truth sample."""

    sample_id: str
    task_id: str
    annotator: str
    central_node: str
    query: str
    annotations: dict[str, str]  # class_name -> "required" | "useful" | "irrelevant"
    status: str
    project: str  # inferred from central_node
    raw: dict[str, Any]  # original row

    def is_annotated(self) -> bool:
        return bool(self.annotations)

    def is_finalized(self) -> bool:
        return self.status.lower() == "finalized"


# Map project name -> diagram filename in full_diagrams_fixed_generic/
_PROJECT_PATTERNS = [
    ("hadoop", "hadoop.json"),
    ("flink", "flink.json"),
    ("ghidra", "ghidra.json"),
    ("dbeaver", "dbeaver.json"),
    ("nd4j", "deeplearning4j.json"),
    ("deeplearning4j", "deeplearning4j.json"),
    ("activiti", "Activiti.json"),
    ("lmax.disruptor", "disruptor.json"),
    ("thingsboard", "thingsboard.json"),
]


def infer_project(central_node: str) -> str:
    """Infer project slug (matching diagram file stem) from a class path."""
    lower = central_node.lower()
    for pattern, fname in _PROJECT_PATTERNS:
        if pattern in lower:
            return Path(fname).stem
    return "unknown"


def infer_diagram_filename(central_node: str) -> str:
    """Return the diagram JSON filename corresponding to a central node."""
    lower = central_node.lower()
    for pattern, fname in _PROJECT_PATTERNS:
        if pattern in lower:
            return fname
    return ""


def _parse_annotations_field(raw: str) -> dict[str, str]:
    """Parse the entity_annotations column (JSON string)."""
    if not raw or raw == "{}":
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: handle possible double-escaped quotes.
        fixed = raw.replace('""', '"')
        return json.loads(fixed)


# Label priority for union-merging: higher wins
_LABEL_PRIORITY = {"required": 3, "useful": 2, "irrelevant": 1}


def _merge_annotations(dicts: list[dict[str, str]]) -> dict[str, str]:
    """Union-merge several annotator label dicts.

    For each class, keep the strongest label across annotators:
    required > useful > irrelevant.
    """
    merged: dict[str, str] = {}
    for d in dicts:
        for cls, lbl in d.items():
            prev = merged.get(cls)
            if prev is None or _LABEL_PRIORITY.get(lbl, 0) > _LABEL_PRIORITY.get(
                prev, 0
            ):
                merged[cls] = lbl
    return merged


def load_annotations(
    csv_path: str | Path,
    finalized_only: bool = True,
    annotated_only: bool = True,
    merge_annotators: bool = True,
) -> list[AnnotationSample]:
    """Load and filter annotation samples from the CSV file.

    Args:
        csv_path: Path to annotations.csv.
        finalized_only: Keep only rows with status == 'Finalized'.
        annotated_only: Keep only rows with non-empty entity_annotations.
        merge_annotators: Merge multiple rows sharing the same sample_id by
            unioning their annotations (label priority: required > useful >
            irrelevant). If False, return one sample per (sample_id, annotator).

    Returns:
        List of AnnotationSample.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {path}")

    raw_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations = _parse_annotations_field(row.get("entity_annotations", ""))
            status = row.get("status", "") or ""
            if annotated_only and not annotations:
                continue
            if finalized_only and status.lower() != "finalized":
                continue
            raw_rows.append({**row, "_parsed_annotations": annotations})

    if not merge_annotators:
        samples = []
        for row in raw_rows:
            central_node = row.get("central_node", "") or ""
            samples.append(
                AnnotationSample(
                    sample_id=row.get("sample_id", "") or row.get("_id", ""),
                    task_id=row.get("task_id", ""),
                    annotator=row.get("annotator", ""),
                    central_node=central_node,
                    query=row.get("query", ""),
                    annotations=row["_parsed_annotations"],
                    status=row.get("status", ""),
                    project=infer_project(central_node),
                    raw=row,
                )
            )
        return samples

    # Group by sample_id and merge
    by_sid: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        sid = row.get("sample_id", "") or row.get("_id", "")
        by_sid.setdefault(sid, []).append(row)

    samples: list[AnnotationSample] = []
    for sid, rows in by_sid.items():
        first = rows[0]
        merged_annotations = _merge_annotations(
            [r["_parsed_annotations"] for r in rows]
        )
        annotators = sorted(
            {r.get("annotator", "") for r in rows if r.get("annotator")}
        )
        central_node = first.get("central_node", "") or ""
        samples.append(
            AnnotationSample(
                sample_id=sid,
                task_id=first.get("task_id", ""),
                annotator=",".join(annotators),
                central_node=central_node,
                query=first.get("query", ""),
                annotations=merged_annotations,
                status=first.get("status", ""),
                project=infer_project(central_node),
                raw=first,
            )
        )
    return samples


def group_samples_by_project(
    samples: list[AnnotationSample],
) -> dict[str, list[AnnotationSample]]:
    """Group samples by inferred project."""
    out: dict[str, list[AnnotationSample]] = {}
    for s in samples:
        out.setdefault(s.project, []).append(s)
    return out
