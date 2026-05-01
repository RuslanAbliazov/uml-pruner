"""Approach #4: human-like agent (anchors -> centrality expansion -> prune).

Implementation lives in :mod:`runner`. See ``README.md`` in
``scripts/approaches/human_like_agent/`` for the user-facing description.
"""

from src.approaches.human_like_agent.runner import NAME, build_runner

__all__ = ["NAME", "build_runner"]
