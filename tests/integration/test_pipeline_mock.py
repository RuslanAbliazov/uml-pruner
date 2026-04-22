"""Integration test using a mocked LLM to verify the 2-stage pipeline.

Scenarios:
1. Normal budget: pipeline runs Stage 1 + Stage 2 without splitting.
2. Tiny budget: autosplit kicks in on both stages; pipeline still returns
   sensible output without raising.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.client import LLMClient, LLMResponse
from src.pipeline.pipeline import PipelineConfig, run_pipeline
from src.utils.io import load_diagram


def _make_mock_call(diagram):
    """Mock LLM that returns plausible Stage 1 and Stage 2 JSON responses."""
    all_node_ids = [n["node_id"] for n in diagram["nodes"]]
    all_packages: set[str] = set()
    for nid in all_node_ids:
        parts = nid.split(".")
        if len(parts) >= 3:
            all_packages.add(".".join(parts[:3]))

    async def mock_call(system_prompt, user_prompt, json_mode=True, max_tokens=None):
        if "Packages in this batch" in user_prompt:
            pkgs = sorted(all_packages)
            content = json.dumps({"high": pkgs[: max(1, len(pkgs) // 2)], "medium": []})
        elif "Classes in this batch" in user_prompt:
            present = [nid for nid in all_node_ids if f'"{nid}"' in user_prompt]
            required = present[: max(1, len(present) // 4)]
            useful = present[len(required) : len(required) + 5]
            content = json.dumps({"required": required, "useful": useful})
        else:
            content = json.dumps({"required": [], "useful": []})
        return LLMResponse(
            content=content, input_tokens=100, output_tokens=50, model="mock"
        )

    return mock_call


async def _run_with_mock(diagram, cfg):
    client = LLMClient(api_key="dummy")
    client.call = AsyncMock(side_effect=_make_mock_call(diagram))
    return await run_pipeline(
        query="Show the main classes involved in the Disruptor ring buffer pattern",
        diagram=diagram,
        llm_client=client,
        cfg=cfg,
    )


async def test_normal_run():
    diagram_path = PROJECT_ROOT / "full_diagrams_fixed_generic" / "disruptor.json"
    diagram = load_diagram(diagram_path)
    cfg = PipelineConfig(
        stage1_batch_size=50,
        stage1_parallel=2,
        stage2_batch_size=100,
        stage2_parallel=2,
        stage2_max_output=200,
    )
    result = await _run_with_mock(diagram, cfg)

    meta = result["metadata"]
    assert meta["original_node_count"] == len(diagram["nodes"])
    assert meta["filtered_node_count"] > 0
    assert meta["stage_sizes"]["stage1_survivors"] > 0
    assert meta["stage_sizes"]["stage2_total"] == meta["filtered_node_count"]

    kept_ids = {n["node_id"] for n in result["nodes"]}
    for e in result["edges"]:
        assert e["node_id_from"] in kept_ids
        assert e["node_id_to"] in kept_ids

    print("PASS: normal run")
    print(f"  Nodes: {len(diagram['nodes'])} -> {meta['filtered_node_count']}")
    print(
        f"  REQUIRED={meta['stage_sizes']['stage2_required']}, "
        f"USEFUL={meta['stage_sizes']['stage2_useful']}"
    )


async def test_overflow_triggers_autosplit():
    diagram_path = PROJECT_ROOT / "full_diagrams_fixed_generic" / "disruptor.json"
    diagram = load_diagram(diagram_path)

    # Tiny window forces autosplit on both Stage 1 and Stage 2.
    cfg = PipelineConfig(
        stage1_batch_size=50,
        stage1_parallel=2,
        stage2_batch_size=100,
        stage2_parallel=2,
        stage2_max_output=200,
        context_window=2_000,
        output_reserve=300,
        safety_margin=100,
        max_split_depth=8,
    )
    result = await _run_with_mock(diagram, cfg)

    meta = result["metadata"]
    assert meta["filtered_node_count"] > 0
    print("PASS: overflow run (autosplit)")
    print(f"  Nodes: {len(diagram['nodes'])} -> {meta['filtered_node_count']}")
    print(
        f"  REQUIRED={meta['stage_sizes']['stage2_required']}, "
        f"USEFUL={meta['stage_sizes']['stage2_useful']}"
    )


async def main():
    await test_normal_run()
    print()
    await test_overflow_triggers_autosplit()


if __name__ == "__main__":
    asyncio.run(main())
