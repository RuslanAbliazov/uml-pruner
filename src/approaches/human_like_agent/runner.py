"""Runner for human_like_agent approach using MCP server for graph navigation."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

from src.core.io import load_diagram
from src.core.logger import get_logger
from src.eval.annotations import diagram_filename_for_repo
from src.llm.client import LLMClient

from .settings import HumanLikeAgentSettings

logger = get_logger(__name__)


class HumanLikeAgentRunner:
    """Runner for human_like_agent approach.
    
    Uses MCP server to provide graph navigation tools to an LLM agent.
    """

    def __init__(self, settings: HumanLikeAgentSettings, llm: LLMClient):
        """Initialize runner.
        
        Args:
            settings: Validated settings from config
            llm: LLM client (for settings compatibility, but we use OpenAI directly for tools)
        """
        self.settings = settings
        self.llm = llm
        
        # We'll use OpenAI client directly for tool calling
        self.openai_client = AsyncOpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url or None,
            timeout=settings.llm.timeout,
        )
        
        # Load prompts
        prompts_dir = Path(__file__).parent / "prompts"
        with open(prompts_dir / "agent_system.txt") as f:
            self.system_prompt_template = f.read()
        with open(prompts_dir / "agent_user.txt") as f:
            self.user_prompt_template = f.read()

    async def run_async(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Run agent on a single sample.
        
        Args:
            sample: Dataset row with 'query', 'sample_id', 'repo', etc.
            
        Returns:
            Dict with 'required' and 'useful' node lists
        """
        sample_id = sample["sample_id"]
        query = sample["query"]
        repo = sample["repo"]
        
        logger.info(f"[{sample_id}] Running human_like_agent for repo={repo}")
        
        # Load anchors from stage2
        anchors = await self._load_anchors(sample)
        if not anchors:
            logger.warning(f"[{sample_id}] No anchors found, returning empty result")
            return {"required": [], "useful": []}
        
        logger.info(f"[{sample_id}] Loaded {len(anchors)} anchors: {anchors}")
        
        # Load graph data
        diagram_filename = diagram_filename_for_repo(repo)
        diagram_path = Path("data/diagrams_normalized") / diagram_filename
        graph_data = load_diagram(diagram_path)
        
        # Start MCP server
        server_script = Path(__file__).parent / "mcp_server.py"
        
        # Run agent with MCP tools
        result = await self._run_agent_with_mcp(
            query=query,
            anchors=anchors,
            graph_data=graph_data,
            server_script=server_script,
            sample_id=sample_id,
        )
        
        logger.info(
            f"[{sample_id}] Agent returned {len(result.get('required', []))} required, "
            f"{len(result.get('useful', []))} useful"
        )
        
        return result

    def run(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Synchronous wrapper for run_async."""
        return asyncio.run(self.run_async(sample))

    async def _load_anchors(self, sample: dict[str, Any]) -> list[str]:
        """Load anchor classes from stage2 output.
        
        Args:
            sample: Dataset row
            
        Returns:
            List of anchor class node_ids
        """
        sample_id = sample["sample_id"]
        repo = sample["repo"]
        
        # Try to load from data/stage2_anchors/
        stage2_dir = Path("data/stage2_anchors")
        anchor_file = stage2_dir / f"{repo}__{sample_id}.json"
        
        if not anchor_file.exists():
            logger.warning(
                f"[{sample_id}] Anchor file not found: {anchor_file}. "
                "Run anchor_neighbors approach first or provide anchors manually."
            )
            return []
        
        with open(anchor_file) as f:
            data = json.load(f)
        
        return data.get("anchors", [])

    async def _run_agent_with_mcp(
        self,
        query: str,
        anchors: list[str],
        graph_data: dict[str, Any],
        server_script: Path,
        sample_id: str,
    ) -> dict[str, Any]:
        """Run agent with MCP server providing graph navigation tools.
        
        Args:
            query: User query
            anchors: List of anchor class node_ids
            graph_data: Graph JSON data
            server_script: Path to mcp_server.py
            sample_id: Sample identifier for logging
            
        Returns:
            Dict with 'required' and 'useful' node lists
        """
        # Create temp file for graph data
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as tmp:
            json.dump(graph_data, tmp)
            graph_file = tmp.name
        
        try:
            # Start MCP server as subprocess
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[str(server_script), graph_file],
                env=None,
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Get available tools from MCP server
                    tools_response = await session.list_tools()
                    mcp_tools = tools_response.tools
                    
                    # Convert MCP tools to OpenAI format
                    openai_tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema,
                            },
                        }
                        for tool in mcp_tools
                    ]
                    
                    # Prepare prompts
                    system_prompt = self.system_prompt_template.format(
                        max_steps=self.settings.max_steps
                    )
                    user_prompt = self.user_prompt_template.format(
                        query=query,
                        anchors=json.dumps(anchors, indent=2),
                        max_steps=self.settings.max_steps,
                    )
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    
                    # Run agent loop with tool calling
                    step_count = 0
                    
                    while step_count < self.settings.max_steps:
                        # Call LLM
                        response = await self.openai_client.chat.completions.create(
                            model=self.settings.llm.model,
                            messages=messages,
                            tools=openai_tools,
                            temperature=self.settings.llm.temperature,
                            max_tokens=self.settings.llm.max_tokens,
                        )
                        
                        message = response.choices[0].message
                        
                        # Add assistant message to history
                        messages.append({
                            "role": "assistant",
                            "content": message.content,
                            "tool_calls": message.tool_calls if message.tool_calls else None,
                        })
                        
                        # Check if agent is done (no tool calls)
                        if not message.tool_calls:
                            # Agent returned final answer
                            content = message.content or ""
                            try:
                                result = json.loads(content)
                                if "required" in result and "useful" in result:
                                    return result
                                else:
                                    logger.warning(
                                        f"[{sample_id}] Agent returned invalid JSON "
                                        f"(missing required/useful): {content[:200]}"
                                    )
                                    return {"required": [], "useful": []}
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"[{sample_id}] Agent returned non-JSON: {content[:200]}"
                                )
                                return {"required": [], "useful": []}
                        
                        # Execute tool calls via MCP
                        for tool_call in message.tool_calls:
                            step_count += 1
                            if step_count > self.settings.max_steps:
                                logger.warning(
                                    f"[{sample_id}] Reached max_steps={self.settings.max_steps}, "
                                    "stopping agent"
                                )
                                break
                            
                            function_name = tool_call.function.name
                            function_args = json.loads(tool_call.function.arguments)
                            
                            logger.debug(
                                f"[{sample_id}] Step {step_count}: {function_name}({function_args})"
                            )
                            
                            # Call MCP tool
                            tool_result = await session.call_tool(
                                function_name, function_args
                            )
                            
                            # Extract text content from MCP result
                            result_text = ""
                            if hasattr(tool_result, 'content'):
                                for content_item in tool_result.content:
                                    if hasattr(content_item, 'text'):
                                        result_text += content_item.text
                            
                            # Add tool result to messages
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result_text,
                            })
                    
                    # If we exit loop due to max_steps, ask agent for final answer
                    logger.warning(
                        f"[{sample_id}] Max steps reached, requesting final answer"
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            "You have reached the maximum number of tool calls. "
                            "Please return your final answer now as JSON with "
                            "'required' and 'useful' lists."
                        ),
                    })
                    
                    final_response = await self.openai_client.chat.completions.create(
                        model=self.settings.llm.model,
                        messages=messages,
                        temperature=self.settings.llm.temperature,
                        max_tokens=self.settings.llm.max_tokens,
                    )
                    
                    final_content = final_response.choices[0].message.content or ""
                    try:
                        result = json.loads(final_content)
                        if "required" in result and "useful" in result:
                            return result
                    except json.JSONDecodeError:
                        pass
                    
                    logger.error(
                        f"[{sample_id}] Agent failed to return valid JSON after max_steps"
                    )
                    return {"required": [], "useful": []}
                    
        finally:
            # Clean up temp file
            import os
            try:
                os.unlink(graph_file)
            except Exception:
                pass
