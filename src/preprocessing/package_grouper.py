"""Group UML nodes by their top-level or configurable-depth package prefix."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _get_package_prefix(node_id: str, depth: int) -> str:
    """Return the first `depth` dot-separated segments of node_id.

    Example:
        'com.lmax.disruptor.examples.ShutdownOnError.Handler' with depth=3
        -> 'com.lmax.disruptor'
    """
    parts = node_id.split(".")
    if len(parts) <= depth:
        # If there's no package at all, return everything except last (class name)
        return ".".join(parts[:-1]) or parts[0]
    return ".".join(parts[:depth])


def _choose_package_depth(nodes: list[dict[str, Any]], target_groups: int = 60) -> int:
    """Heuristically pick a package-depth that produces roughly `target_groups`.

    Tries depths 2..6 and picks the one whose group count is closest (but not
    exceeding target_groups * 3).
    """
    best_depth = 3
    best_score = float("inf")
    for depth in range(2, 7):
        prefixes = {_get_package_prefix(n["node_id"], depth) for n in nodes}
        count = len(prefixes)
        # Score: penalize both too-few and too-many groups
        score = abs(count - target_groups)
        if score < best_score:
            best_score = score
            best_depth = depth
    return best_depth


def group_by_package(
    nodes: list[dict[str, Any]],
    depth: int | None = None,
    target_groups: int = 60,
) -> dict[str, list[dict[str, Any]]]:
    """Group nodes by their package prefix.

    Args:
        nodes: List of node dicts (must have 'node_id').
        depth: Package depth (number of dot-segments). If None, auto-chosen.
        target_groups: Target number of groups when auto-choosing depth.

    Returns:
        Dict mapping package_prefix -> list of nodes.
    """
    if depth is None:
        depth = _choose_package_depth(nodes, target_groups=target_groups)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        prefix = _get_package_prefix(node["node_id"], depth)
        groups[prefix].append(node)

    return dict(groups)


def summarize_packages(
    groups: dict[str, list[dict[str, Any]]],
    samples_per_package: int = 8,
) -> list[dict[str, Any]]:
    """Build compact package summaries for Stage 1 prompt.

    Each summary contains: name, count, samples (short class names).
    """
    summaries = []
    for name, members in groups.items():
        samples = []
        for node in members[:samples_per_package]:
            short = node["node_id"].rsplit(".", 1)[-1]
            samples.append(short)
        summaries.append(
            {
                "name": name,
                "count": len(members),
                "samples": samples,
            }
        )
    # Stable order: largest packages first (they are more likely to matter)
    summaries.sort(key=lambda p: -p["count"])
    return summaries
