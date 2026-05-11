"""MCP server providing graph navigation tools for the agent.

Implements Model Context Protocol server with tools for exploring UML class diagrams.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server


class GraphNavigationServer:
    """MCP server wrapping a networkx graph with navigation tools."""

    def __init__(self, graph_data: dict[str, Any]):
        """Initialize server with graph data.
        
        Args:
            graph_data: Dict with 'nodes' and 'edges' keys (from JSON diagram)
        """
        self.graph_data = graph_data
        self.nodes_by_id = {node["node_id"]: node for node in graph_data["nodes"]}
        
        # Build adjacency structure for fast neighbor lookup
        # outgoing[A] = [(B, edge_type)] means A -> B
        # incoming[B] = [(A, edge_type)] means A -> B
        self.outgoing: dict[str, list[tuple[str, str]]] = {}
        self.incoming: dict[str, list[tuple[str, str]]] = {}
        
        for edge in graph_data["edges"]:
            src = edge["node_id_from"]
            tgt = edge["node_id_to"]
            edge_type = edge["description"]
            
            self.outgoing.setdefault(src, []).append((tgt, edge_type))
            self.incoming.setdefault(tgt, []).append((src, edge_type))
        
        self.server = Server("graph-navigation")
        self._register_handlers()

    def _register_handlers(self):
        """Register MCP tool handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available tools."""
            return [
                types.Tool(
                    name="get_node_details",
                    description=(
                        "Get detailed information about a specific class/interface node. "
                        "Returns the complete JSON representation including type, name, "
                        "methods, params (fields), and description."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "node_id": {
                                "type": "string",
                                "description": "Fully qualified class name (e.g., 'org.example.MyClass')",
                            }
                        },
                        "required": ["node_id"],
                    },
                ),
                types.Tool(
                    name="get_neighbors",
                    description=(
                        "Get neighbors of a node in the graph. Returns separate lists for "
                        "incoming and outgoing edges, each with target node_id and edge type. "
                        "Edge types: 'Inheritance' (A inherits from B), 'Dependency' (A uses B), "
                        "'Association' (A has B as field/parameter)."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "node_id": {
                                "type": "string",
                                "description": "Fully qualified class name",
                            },
                            "edge_type": {
                                "type": "string",
                                "description": "Optional filter: 'Inheritance', 'Dependency', or 'Association'",
                            },
                        },
                        "required": ["node_id"],
                    },
                ),
                types.Tool(
                    name="get_edge_details",
                    description=(
                        "Get information about the edge between two nodes. "
                        "Returns edge type and direction."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source_id": {
                                "type": "string",
                                "description": "Source node_id",
                            },
                            "target_id": {
                                "type": "string",
                                "description": "Target node_id",
                            },
                        },
                        "required": ["source_id", "target_id"],
                    },
                ),
                types.Tool(
                    name="search_nodes",
                    description=(
                        "Search for nodes by name pattern (case-insensitive substring match). "
                        "Returns list of matching node_ids."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "Search pattern (substring, case-insensitive)",
                            }
                        },
                        "required": ["pattern"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict | None
        ) -> list[types.TextContent]:
            """Handle tool calls."""
            if arguments is None:
                arguments = {}

            if name == "get_node_details":
                return await self._get_node_details(arguments)
            elif name == "get_neighbors":
                return await self._get_neighbors(arguments)
            elif name == "get_edge_details":
                return await self._get_edge_details(arguments)
            elif name == "search_nodes":
                return await self._search_nodes(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _get_node_details(self, args: dict) -> list[types.TextContent]:
        """Get complete node information."""
        node_id = args["node_id"]
        node = self.nodes_by_id.get(node_id)
        
        if node is None:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({"error": f"Node not found: {node_id}"}),
                )
            ]
        
        return [
            types.TextContent(
                type="text",
                text=json.dumps(node, indent=2),
            )
        ]

    async def _get_neighbors(self, args: dict) -> list[types.TextContent]:
        """Get neighbors with edge types and directions."""
        node_id = args["node_id"]
        edge_type_filter = args.get("edge_type")
        
        if node_id not in self.nodes_by_id:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({"error": f"Node not found: {node_id}"}),
                )
            ]
        
        # Outgoing edges: node_id -> target
        outgoing_edges = []
        for target, edge_type in self.outgoing.get(node_id, []):
            if edge_type_filter is None or edge_type == edge_type_filter:
                outgoing_edges.append({"node_id": target, "edge_type": edge_type})
        
        # Incoming edges: source -> node_id
        incoming_edges = []
        for source, edge_type in self.incoming.get(node_id, []):
            if edge_type_filter is None or edge_type == edge_type_filter:
                incoming_edges.append({"node_id": source, "edge_type": edge_type})
        
        result = {
            "node_id": node_id,
            "outgoing": outgoing_edges,
            "incoming": incoming_edges,
        }
        
        return [
            types.TextContent(
                type="text",
                text=json.dumps(result, indent=2),
            )
        ]

    async def _get_edge_details(self, args: dict) -> list[types.TextContent]:
        """Get edge information between two nodes."""
        source_id = args["source_id"]
        target_id = args["target_id"]
        
        # Find edge from source to target
        edge_type = None
        for tgt, etype in self.outgoing.get(source_id, []):
            if tgt == target_id:
                edge_type = etype
                break
        
        if edge_type is None:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({
                        "error": f"No edge found from {source_id} to {target_id}"
                    }),
                )
            ]
        
        result = {
            "source_id": source_id,
            "target_id": target_id,
            "edge_type": edge_type,
            "direction": "outgoing",
        }
        
        return [
            types.TextContent(
                type="text",
                text=json.dumps(result, indent=2),
            )
        ]

    async def _search_nodes(self, args: dict) -> list[types.TextContent]:
        """Search nodes by name pattern."""
        pattern = args["pattern"].lower()
        
        matching_ids = [
            node_id
            for node_id in self.nodes_by_id
            if pattern in node_id.lower()
        ]
        
        result = {
            "pattern": args["pattern"],
            "matches": matching_ids,
            "count": len(matching_ids),
        }
        
        return [
            types.TextContent(
                type="text",
                text=json.dumps(result, indent=2),
            )
        ]

    async def run(self):
        """Run the MCP server on stdio."""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


def create_server(graph_data: dict[str, Any]) -> GraphNavigationServer:
    """Create MCP server instance for graph navigation.
    
    Args:
        graph_data: Dict with 'nodes' and 'edges' from normalized diagram JSON
        
    Returns:
        GraphNavigationServer instance ready to run
    """
    return GraphNavigationServer(graph_data)


async def run_server(graph_data: dict[str, Any]):
    """Run MCP server for graph navigation.
    
    This is the main entry point for starting the server process.
    
    Args:
        graph_data: Dict with 'nodes' and 'edges' from normalized diagram JSON
    """
    server = create_server(graph_data)
    await server.run()


if __name__ == "__main__":
    # For testing: load a sample graph and run server
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MCP server for UML diagram graph navigation"
    )
    parser.add_argument(
        "--diagram_path",
        type=str,
        help="Path to normalized diagram JSON file",
    )
    args = parser.parse_args()
    
    with open(args.diagram_path, "r") as f:
        graph_data = json.load(f)
    
    asyncio.run(run_server(graph_data))
