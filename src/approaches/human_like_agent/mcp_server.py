"""Minimal stateful MCP server for UML graph exploration.

The agent builds a working graph incrementally by:
1. Starting with anchors
2. Previewing neighbors
3. Adding selected nodes to working memory
4. Marking nodes as unrecognized/useful/required
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

# Limits (configurable via args)
# MAX_WORKING_GRAPH_SIZE = 30
# MAX_PREVIEW_NEIGHBORS = 50


class GraphNavigationServer:
    """Stateful MCP server with working memory for incremental graph building."""

    def __init__(self, graph_data: dict[str, Any]):
        self.full_graph = graph_data
        self.nodes_by_id = {n["node_id"]: n for n in graph_data["nodes"]}
        
        # Build adjacency lists
        self.outgoing: dict[str, list[tuple[str, str]]] = {}
        self.incoming: dict[str, list[tuple[str, str]]] = {}
        for edge in graph_data["edges"]:
            src, tgt = edge["node_id_from"], edge["node_id_to"]
            edge_type = edge["description"]
            self.outgoing.setdefault(src, []).append((tgt, edge_type))
            self.incoming.setdefault(tgt, []).append((src, edge_type))
        
        # Working memory (important nodes for answering query)
        self.working_nodes: set[str] = set()  # just node_ids
        
        self.server = Server("graph-navigation")
        self._register_handlers()

    def _register_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name="add_nodes",
                    description="Add important nodes to working graph.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "node_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Node IDs to add"
                            }
                        },
                        "required": ["node_ids"],
                    },
                ),
                types.Tool(
                    name="preview_neighbors",
                    description="See neighbors WITHOUT adding to working graph. Returns node IDs and edge types only.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "node_id": {"type": "string"},
                            "direction": {
                                "type": "string",
                                "enum": ["outgoing", "incoming", "both"],
                                "default": "both"
                            },
                        },
                        "required": ["node_id"],
                    },
                ),
                types.Tool(
                    name="get_node_details",
                    description="Get type + methods for nodes. No description field.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "node_ids": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["node_ids"],
                    },
                ),
                types.Tool(
                    name="mark_final_statuses",
                    description="Mark final statuses for ALL nodes in working graph. Call this once when ready to finish.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "required_node_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Core nodes answering the query"
                            },
                            "useful_node_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Supporting nodes providing context"
                            },
                            "irrelevant_node_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Nodes that turned out to be irrelevant (will be excluded from result)",
                                "default": []
                            }
                        },
                        "required": ["required_node_ids", "useful_node_ids"],
                    },
                ),
                types.Tool(
                    name="get_working_graph",
                    description="See current working graph node list.",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
            args = arguments or {}
            
            if name == "add_nodes":
                return await self._add_nodes(args)
            elif name == "preview_neighbors":
                return await self._preview_neighbors(args)
            elif name == "get_node_details":
                return await self._get_node_details(args)
            elif name == "mark_final_statuses":
                return await self._mark_final_statuses(args)
            elif name == "get_working_graph":
                return await self._get_working_graph(args)
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _preview_neighbors(self, args: dict) -> list[types.TextContent]:
        node_id = args["node_id"]
        direction = args.get("direction", "both")
        
        if node_id not in self.nodes_by_id:
            return [types.TextContent(type="text", text=json.dumps({"error": f"Node not found: {node_id}"}))]
        
        result = {"node_id": node_id, "neighbors": {}}
        
        if direction in ("outgoing", "both"):
            neighbors = self.outgoing.get(node_id, [])
            result["neighbors"]["outgoing"] = [
                {"node_id": tgt, "edge_type": etype} 
                for tgt, etype in sorted(neighbors) # [:MAX_PREVIEW_NEIGHBORS]
            ]
        
        if direction in ("incoming", "both"):
            neighbors = self.incoming.get(node_id, [])
            result["neighbors"]["incoming"] = [
                {"node_id": src, "edge_type": etype} 
                for src, etype in sorted(neighbors) # [:MAX_PREVIEW_NEIGHBORS]
            ]
        
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _add_nodes(self, args: dict) -> list[types.TextContent]:
        node_ids = args["node_ids"]
        
        added = []
        not_found = []
        already_present = []
        
        for nid in node_ids:
            if nid not in self.nodes_by_id:
                not_found.append(nid)
            elif nid in self.working_nodes:
                already_present.append(nid)
            else:
                self.working_nodes.add(nid)
                added.append(nid)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "added": added,
                "already_present": already_present,
                "not_found": not_found,
                "working_graph_size": len(self.working_nodes)
            })
        )]

    async def _get_node_details(self, args: dict) -> list[types.TextContent]:
        node_ids = args["node_ids"]
        nodes = []
        
        for nid in node_ids:
            node = self.nodes_by_id.get(nid)
            if node:
                # Return type + methods only (no description)
                nodes.append({
                    "node_id": node["node_id"],
                    "type": node["type"],
                    "methods": node.get("methods", [])
                })
        
        return [types.TextContent(type="text", text=json.dumps({"nodes": nodes}, indent=2))]

    async def _mark_final_statuses(self, args: dict) -> list[types.TextContent]:
        required = set(args["required_node_ids"])
        useful = set(args["useful_node_ids"])
        irrelevant = set(args.get("irrelevant_node_ids", []))
        
        # Validate all marked nodes are in working graph
        all_marked = required | useful | irrelevant
        not_in_working = []
        for nid in all_marked:
            if nid not in self.working_nodes:
                not_in_working.append(nid)
        
        if not_in_working:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "error": "Some nodes not in working graph",
                    "not_in_working_graph": not_in_working
                })
            )]
        
        # Check all working nodes are marked
        unmarked = self.working_nodes - all_marked
        if unmarked:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "error": "Some working nodes not marked",
                    "unmarked_nodes": list(unmarked),
                    "hint": "All nodes must be marked as required, useful, or irrelevant"
                })
            )]
        
        # Build final result (exclude irrelevant)
        final_nodes = required | useful
        nodes = [self.nodes_by_id[nid] for nid in final_nodes]
        edges = []
        for edge in self.full_graph["edges"]:
            if edge["node_id_from"] in final_nodes and edge["node_id_to"] in final_nodes:
                edges.append(edge)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "nodes": nodes,
                "edges": edges,
                "required_node_ids": list(required),
                "useful_node_ids": list(useful)
            }, indent=2)
        )]

    async def _get_working_graph(self, args: dict) -> list[types.TextContent]:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "node_ids": list(self.working_nodes),
                "total_nodes": len(self.working_nodes)
            }, indent=2)
        )]

    async def run(self):
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", required=True, help="Path to normalized diagram JSON")
    args = parser.parse_args()
    
    graph_path = Path(args.repo_path)
    if not graph_path.exists():
        print(f"ERROR: Graph file not found: {graph_path}", file=sys.stderr)
        sys.exit(1)
    
    with open(graph_path) as f:
        graph_data = json.load(f)
    
    server = GraphNavigationServer(graph_data)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
