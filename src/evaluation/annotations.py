"""Load the consolidated annotation dataset.

This module ONLY reads a dataset CSV produced by `scripts/build_dataset.py`.
It does no merging, voting, filtering, or any other data preparation —
preparation lives exclusively in the build script.

Expected CSV schema:
    _id, sample_id, task_id, central_node, repo, query, entity_annotations

`entity_annotations` is a JSON string mapping node_id -> "required" | "useful".
By construction the build script never emits "irrelevant" labels, so the
in-memory dict only contains required/useful entries.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_FIELDS = (
    "_id",
    "sample_id",
    "task_id",
    "central_node",
    "repo",
    "query",
    "entity_annotations",
)


# ----------------------------------------------------------------------------
# Repo -> diagram filename mapping (the only "metadata" we still keep here).
# Build-time uses the diagram's contents; runtime needs the inverse: given a
# sample's `repo` field, which JSON file should the pipeline load?
# ----------------------------------------------------------------------------

_REPO_TO_DIAGRAM = {
    "apache/hadoop": "hadoop.json",
    "apache/flink": "flink.json",
    "NationalSecurityAgency/ghidra": "ghidra.json",
    "dbeaver/dbeaver": "dbeaver.json",
    "deeplearning4j/deeplearning4j": "deeplearning4j.json",
    "Activiti/Activiti": "Activiti.json",
    "LMAX-Exchange/disruptor": "disruptor.json",
    "thingsboard/thingsboard": "thingsboard.json",
}


def diagram_filename_for_repo(repo: str) -> str:
    """Return the diagram filename associated with `repo`.

    Falls back to "<repo-stem>.json" derived from the second path segment of
    the slug (e.g. "owner/name" -> "name.json"), so unmapped projects still
    work as long as the diagram file is named after the repo.
    """
    if repo in _REPO_TO_DIAGRAM:
        return _REPO_TO_DIAGRAM[repo]
    if "/" in repo:
        return f"{repo.split('/', 1)[1]}.json"
    return f"{repo}.json"


# ----------------------------------------------------------------------------
# Sample dataclass
# ----------------------------------------------------------------------------


@dataclass
class AnnotationSample:
    """One consolidated dataset row."""

    _id: str
    sample_id: str
    task_id: str
    central_node: str
    repo: str
    query: str
    annotations: dict[str, str]  # node_id -> "required" | "useful"

    @property
    def diagram_filename(self) -> str:
        return diagram_filename_for_repo(self.repo)

    @property
    def project(self) -> str:
        """Diagram filename stem (e.g. "hadoop"). Kept for backwards compat."""
        return Path(self.diagram_filename).stem

    def to_dict(self) -> dict[str, Any]:
        return {
            "_id": self._id,
            "sample_id": self.sample_id,
            "task_id": self.task_id,
            "central_node": self.central_node,
            "repo": self.repo,
            "query": self.query,
            "entity_annotations": dict(self.annotations),
        }


# ----------------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------------


def _parse_annotations_field(raw: str) -> dict[str, str]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(raw.replace('""', '"'))


def load_dataset(csv_path: str | Path) -> list[AnnotationSample]:
    """Read the dataset CSV produced by `scripts/build_dataset.py`.

    Args:
        csv_path: Path to the consolidated dataset CSV.

    Returns:
        List of AnnotationSample. The dataset is assumed already filtered,
        voted, and ready to use; this function does NOT modify it.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found: {path}. "
            f"Run scripts/build_dataset.py first to generate it."
        )

    samples: list[AnnotationSample] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in DATASET_FIELDS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"Dataset CSV {path} is missing required columns: {missing}. "
                f"Expected: {list(DATASET_FIELDS)}"
            )
        for row in reader:
            samples.append(
                AnnotationSample(
                    _id=row["_id"],
                    sample_id=row["sample_id"],
                    task_id=row["task_id"],
                    central_node=row["central_node"],
                    repo=row["repo"],
                    query=row["query"],
                    annotations=_parse_annotations_field(
                        row.get("entity_annotations", "")
                    ),
                )
            )
    return samples


def group_samples_by_repo(
    samples: list[AnnotationSample],
) -> dict[str, list[AnnotationSample]]:
    """Group samples by repo slug."""
    out: dict[str, list[AnnotationSample]] = {}
    for s in samples:
        out.setdefault(s.repo, []).append(s)
    return out
