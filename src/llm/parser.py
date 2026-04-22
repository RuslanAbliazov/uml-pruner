"""Parse JSON responses from LLM, resilient to minor formatting issues."""

from __future__ import annotations

import json
import re
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_json_response(content: str) -> dict[str, Any] | list[Any]:
    """Parse a string that should contain JSON.

    Tries direct parsing first, then strips code fences and re-parses.

    Raises:
        ValueError: if no valid JSON can be recovered.
    """
    if not content or not content.strip():
        raise ValueError("Empty LLM response")

    text = content.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to strip code fences ```json ... ```
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Find first { or [ and matching end
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    logger.error("Failed to parse JSON. Raw content (first 500 chars): %s", text[:500])
    raise ValueError("Could not parse JSON from LLM response")
