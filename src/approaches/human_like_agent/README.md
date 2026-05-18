# Human-Like Agent: Incremental Graph Building

Minimal agent-based approach where LLM builds a working graph step-by-step.

## Core Idea

The agent starts with one anchor and incrementally adds nodes to a working graph (max 30 nodes):

1. **Pick anchor** → `add_nodes([anchor_id])`
2. **Preview neighbors** → See IDs + edge types (cheap)
3. **Add selected** → Only promising nodes enter working graph
4. **Mark status** → unrecognized → useful/required
5. **Repeat** until query answered
6. **Get result** → Returns required + useful nodes

## MCP Tools (6 tools)

### 1. `add_nodes([node_ids], status?)`
Add nodes to working graph. Use this to start with anchor or add neighbors. Enforces 30-node limit.

### 2. `preview_neighbors(node_id, direction?)`
See neighbors WITHOUT adding to working graph. Returns only node IDs + edge types (~50 tokens).

**Example:**
```json
{
  "neighbors": {
    "outgoing": [
      {"node_id": "Path", "edge_type": "Association"},
      {"node_id": "Config", "edge_type": "Dependency"}
    ]
  }
}
```

### 3. `get_node_details([node_ids])`
Get type + methods for nodes. No description field.

### 4. `mark_status([node_ids], status)`
Mark as "unrecognized", "useful", or "required".

### 5. `get_working_graph(verbose?)`
Check current state.

### 6. `get_final_result()`
Return pruned diagram. All nodes must be marked (no unrecognized).

## Configuration

In `configs/config.yaml`:

```yaml
approaches:
  human_like_agent:
    max_steps: 40              # Tool call limit
    outputs_dir: "data/results/human_like_agent"
    llm_traces_dir: "data/llm_traces/human_like_agent"
```

MCP server uses `--max-graph-size 30` (see `opencode.json`).

## Usage

```bash
# Run on 1 sample
python src/approaches/human_like_agent/run.py --limit 1

# Specific repo
python src/approaches/human_like_agent/run.py --repo apache/hadoop

# Via generic runner
python scripts/run.py --approach human_like_agent --limit 5
```

## How It Works

```
Agent receives: query + anchors
  ↓
add_nodes([anchor1])  # Start with one anchor
  ↓
Loop:
  preview_neighbors(node_id) → [A, B, C, ...]
  add_nodes([A, C])           # Skip B (looks irrelevant)
  get_node_details([A, C])    # Get methods
  mark_status([A], "required")
  mark_status([C], "useful")
  ↓
get_final_result()
  → {required: [...], useful: [...]}
```

## Key Constraints

- **30 nodes max** in working graph
- **All nodes must be marked** before `get_final_result()`
- **No description field** — judge by name + methods only
- **~40 tool calls** budget

## Example Session

```
[1] add_nodes(["FileSystem"]) → Working: 1 node
[2] preview_neighbors("FileSystem") → See 47 neighbors
[3] add_nodes(["Path", "Config"]) → Working: 3 nodes
[4] get_node_details(["Path"]) → See methods
[5] mark_status(["Path"], "required")
[6] mark_status(["FileSystem"], "required")
[7] mark_status(["Config"], "useful")
[8] preview_neighbors("Path") → See 15 neighbors
... (continue until query answered)
[20] get_final_result() → {required: [...], useful: [...]}
```

## Troubleshooting

**"Working graph size limit exceeded"**
→ Tried to add too many nodes. Be more selective (max 30 nodes).

**"Cannot generate result: unrecognized nodes present"**
→ Call `mark_status()` for all nodes before `get_final_result()`.

**No anchors found**
→ Run `anchor_neighbors` first to generate anchors:
```bash
python scripts/run.py --approach anchor_neighbors --limit 1
```

## Design Philosophy

- **Minimal code**: ~330 lines MCP server, simple prompts
- **6 tools only**: No redundant operations
- **Stateful memory**: Server tracks working graph
- **Incremental**: Preview → Select → Add (not "expand all neighbors")
- **Controlled context**: Only selected nodes enter LLM context
