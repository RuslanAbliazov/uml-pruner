# Human-Like Agent Approach

Agent-based UML diagram exploration using MCP (Model Context Protocol) tools for graph navigation.

## Overview

The `human_like_agent` approach simulates how a human software architect would explore a class diagram to answer a query:

1. **Start with anchor classes** (from `anchor_neighbors` stage 2)
2. **Inspect each anchor** to understand its structure (methods, fields, relationships)
3. **Explore neighbors** by following edges (inheritance, dependencies, associations)
4. **Make decisions** about which classes are relevant:
   - `required`: Core classes directly answering the query
   - `useful`: Supporting classes providing context (interfaces, base classes, key dependencies)

Unlike batch approaches that process the entire graph at once, the agent **interactively navigates** the graph using tools, mimicking human exploration patterns.

## Architecture

**Important:** One MCP server per sample (repo + query). Each sample gets a fresh server instance.

```
┌─────────────────────────────────────────────────────────────┐
│                  HumanLikeAgentRunner                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  For EACH sample:                                     │  │
│  │  1. Load anchors from data/stage2_anchors/           │  │
│  │  2. Load graph from data/diagrams_normalized/        │  │
│  │  3. Start NEW MCP server as subprocess               │  │
│  │  4. Run LLM agent with tools (OpenAI function calling)│  │
│  │  5. Terminate server, cleanup temp file              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ stdio communication (JSON-RPC)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│          MCP Server (subprocess, one per sample)            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Tools:                                               │  │
│  │  • get_node_details(node_id)                         │  │
│  │  • get_neighbors(node_id, edge_type?)                │  │
│  │  • get_edge_details(source_id, target_id)            │  │
│  │  • search_nodes(pattern)                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  Graph loaded in memory for THIS sample:                   │
│  • nodes_by_id = {node_id: node_data}                      │
│  • outgoing = {node_id: [(target, edge_type), ...]}        │
│  • incoming = {node_id: [(source, edge_type), ...]}        │
└─────────────────────────────────────────────────────────────┘

Lifecycle:
  Sample 1 (hadoop) → MCP Server #1 → Agent explores → Server terminates
  Sample 2 (flink)  → MCP Server #2 → Agent explores → Server terminates
  
No state is shared between samples (full isolation).
```

## MCP Tools

The agent has access to 4 tools for exploring the class diagram:

### 1. `get_node_details(node_id)`
Get complete information about a class:
```json
{
  "type": "class",
  "name": "org.example.MyClass",
  "node_id": "org.example.MyClass",
  "methods": [
    "public void doSomething(String arg)",
    "private int calculate()"
  ],
  "params": [
    "private Logger log",
    "private Service service"
  ],
  "description": ""
}
```

### 2. `get_neighbors(node_id, edge_type?)`
Get neighboring classes with relationship types:
```json
{
  "node_id": "org.example.MyClass",
  "outgoing": [
    {"node_id": "org.example.Service", "edge_type": "Association"},
    {"node_id": "org.example.BaseClass", "edge_type": "Inheritance"}
  ],
  "incoming": [
    {"node_id": "org.example.Controller", "edge_type": "Dependency"}
  ]
}
```

**Edge semantics:**
- **Inheritance** (`A -> B`): A inherits from B (extends/implements)
- **Dependency** (`A -> B`): A uses B (local variable, parameter, return type)
- **Association** (`A -> B`): A has B (field, parameter, generic type parameter)

### 3. `get_edge_details(source_id, target_id)`
Get specific edge information:
```json
{
  "source_id": "org.example.MyClass",
  "target_id": "org.example.BaseClass",
  "edge_type": "Inheritance",
  "direction": "outgoing"
}
```

### 4. `search_nodes(pattern)`
Search for classes by name (case-insensitive):
```json
{
  "pattern": "Controller",
  "matches": [
    "org.example.UserController",
    "org.example.AdminController"
  ],
  "count": 2
}
```

## Configuration

Add to `configs/config.yaml`:

```yaml
approaches:
  human_like_agent:
    # Maximum number of tool calls (to control API costs)
    max_steps: 40
    # Output directories
    outputs_dir: "data/results/human_like_agent"
    llm_traces_dir: "data/llm_traces/human_like_agent"
```

## Usage

### Prerequisites

1. **Install MCP dependencies:**
   ```bash
   pip install -r requirements-mcp.txt
   ```

2. **Generate anchor classes** (run `anchor_neighbors` stage 1-2):
   ```bash
   python scripts/run.py --approach anchor_neighbors --limit 5
   ```
   This creates `data/stage2_anchors/<repo>__<sample_id>.json`

### Running

**Via dedicated CLI (recommended):**
```bash
# Basic run
python src/approaches/human_like_agent/run.py --limit 1

# Specific repository
python src/approaches/human_like_agent/run.py --repo apache/hadoop

# With evaluation
python src/approaches/human_like_agent/run.py --repo apache/flink --eval

# Debug specific query
python src/approaches/human_like_agent/run.py \
    --repo apache/hadoop \
    --query "Show file system classes" \
    --verbose
```

**Programmatic usage (for OpenCode dialog or scripts):**
```python
from src.approaches.human_like_agent.runner import HumanLikeAgentRunner
from src.approaches.human_like_agent.settings import load_settings, make_llm_client
from src.core.config import load_config

cfg = load_config("configs/config.yaml")
settings = load_settings(cfg)
llm = make_llm_client(settings.llm)
runner = HumanLikeAgentRunner(settings, llm)

sample = {
    "sample_id": "abc123",
    "repo": "apache/hadoop",
    "query": "Show file system classes"
}

result = await runner.run_async(sample)
# → {"required": [...], "useful": [...]}
```

### Output

Results are written to `data/results/human_like_agent/`:

```
data/results/human_like_agent/
├── <repo>__<sample_id>.json       # Per-sample results
└── evaluation_report.json          # Aggregate metrics (if --eval)
```

**Per-sample output format:**
```json
{
  "required": [
    "org.example.CoreClass1",
    "org.example.CoreClass2"
  ],
  "useful": [
    "org.example.BaseInterface",
    "org.example.HelperClass"
  ]
}
```

## How It Works

### Agent Loop

1. **Initialization:**
   - Load anchor classes from `data/stage2_anchors/`
   - Load full graph from `data/diagrams_normalized/`
   - Start MCP server as subprocess

2. **Agent Exploration** (up to `max_steps` tool calls):
   ```
   Agent receives:
     - System prompt with tool descriptions
     - User query
     - List of anchor classes
   
   Agent loop:
     while steps < max_steps:
       1. Agent decides which tool to call
       2. MCP server executes tool on graph
       3. Result is returned to agent
       4. Agent updates its understanding
       5. Repeat or return final answer
   ```

3. **Final Answer:**
   ```json
   {
     "required": ["ClassName1", "ClassName2", ...],
     "useful": ["ClassName3", "ClassName4", ...]
   }
   ```

### Example Agent Trace

```
[Step 1] get_node_details("org.example.UserService")
  → Returns: methods, fields, type

[Step 2] get_neighbors("org.example.UserService")
  → Returns: outgoing=[UserRepository, User], incoming=[UserController]

[Step 3] get_node_details("org.example.UserRepository")
  → Returns: methods, fields, type

[Step 4] get_neighbors("org.example.UserRepository", "Inheritance")
  → Returns: outgoing=[BaseRepository]

[Step 5] Returns final JSON:
  {
    "required": ["org.example.UserService", "org.example.UserRepository"],
    "useful": ["org.example.User", "org.example.BaseRepository"]
  }
```

## Prompts

Located in `src/approaches/human_like_agent/prompts/`:

- **`agent_system.txt`**: Agent instructions, tool descriptions, edge semantics, output format
- **`agent_user.txt`**: Per-query template with anchors and max_steps reminder

Key instructions:
- Start with anchor classes
- Explore neighbors efficiently (budget = `max_steps` tool calls)
- Classify classes as `required` (core) vs `useful` (supporting)
- Return valid JSON only

## Cost Control

**Important:** LLM function calling can be expensive. The `max_steps` parameter is a **hard limit** to prevent runaway costs.

Typical values:
- `max_steps: 20` — Quick exploration (5-10 classes)
- `max_steps: 40` — Medium exploration (default, 10-20 classes)
- `max_steps: 100` — Deep exploration (20+ classes, expensive)

**Monitoring:**
- Each tool call consumes 1 step
- Agent stops automatically at `max_steps`
- Check LLM traces in `data/llm_traces/human_like_agent/`

## Comparison with Other Approaches

| Approach | Strategy | Pros | Cons |
|----------|----------|------|------|
| `anchor_neighbors` | Batch: anchor + 1-hop neighbors → LLM prune | Fast, predictable cost | Limited to 1-hop, no adaptive exploration |
| `rag_classes_filter` | Batch: RAG top-K → LLM filter | Simple, no graph structure | Misses structural relationships |
| **`human_like_agent`** | **Interactive: agent navigates graph with tools** | **Human-like exploration, adaptive depth** | **Higher cost, variable steps** |

## Dependencies

**Core (already in requirements.txt):**
- `openai>=1.30.0` (for LLM with function calling)
- Standard project dependencies

**MCP (install separately):**
```bash
pip install -r requirements-mcp.txt
```

Or manually:
```bash
pip install mcp>=0.9.0
```

## Troubleshooting

### "No anchors found"
Run `anchor_neighbors` stage 1-2 first:
```bash
python scripts/run.py --approach anchor_neighbors
```

### MCP import error
Install MCP dependencies:
```bash
pip install -r requirements-mcp.txt
```

### Agent hits max_steps
Increase `max_steps` in `configs/config.yaml`:
```yaml
approaches:
  human_like_agent:
    max_steps: 60  # or higher
```

### Diagram not found
Ensure diagrams are normalized:
```bash
python scripts/normalize_diagrams.py
```

## Future Improvements

Potential enhancements:
- **Graph metrics tools**: betweenness centrality, PageRank
- **Multi-hop path finding**: find path between two classes
- **Code context**: integrate source code snippets
- **Caching**: reuse exploration results across similar queries
- **Budget allocation**: dynamic step budgets based on query complexity

## See Also

- `src/approaches/anchor_neighbors/` — Batch approach with anchor + neighbors
- `scripts/run.py` — Generic approach runner
- `configs/config.yaml` — Configuration reference
