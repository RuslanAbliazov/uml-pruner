# Human-Like Agent: Архитектура и взаимодействие

## Lifecycle: Обработка одного сэмпла

```
┌─────────────────────────────────────────────────────────────────┐
│                        scripts/run.py                           │
│  • Загружает dataset.csv                                        │
│  • Для каждого сэмпла вызывает runner.run(sample)              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    sample = {
                      "repo": "apache/hadoop",
                      "query": "Show file system classes",
                      "sample_id": "abc123"
                    }
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              HumanLikeAgentRunner.run_async(sample)             │
│                                                                 │
│  1. Загружает anchors из:                                       │
│     data/stage2_anchors/apache/hadoop__abc123.json             │
│     → anchors = ["org.apache.hadoop.fs.FileSystem"]            │
│                                                                 │
│  2. Загружает граф из:                                          │
│     data/diagrams_normalized/hadoop.json                       │
│     → graph_data = {nodes: [...], edges: [...]}                │
│                                                                 │
│  3. Создаёт временный файл:                                     │
│     /tmp/tmpXXXXX.json ← graph_data (для этого сэмпла)         │
│                                                                 │
│  4. Запускает MCP-сервер как subprocess:                        │
│     python mcp_server.py /tmp/tmpXXXXX.json                    │
│                                                                 │
│  5. Агент (LLM) работает с MCP-сервером через stdio            │
│                                                                 │
│  6. Возвращает результат:                                       │
│     {required: [...], useful: [...]}                            │
│                                                                 │
│  7. MCP-сервер завершается (context manager)                    │
│  8. Временный файл удаляется                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Детальный процесс взаимодействия

### Шаг 1: Инициализация (для каждого сэмпла)

```python
# runner.py: run_async(sample)
sample_id = sample["sample_id"]  # "abc123"
repo = sample["repo"]             # "apache/hadoop"
query = sample["query"]           # "Show file system classes"

# Загружаем anchors (результат anchor_neighbors stage 2)
anchors = ["org.apache.hadoop.fs.FileSystem"]

# Загружаем граф для ЭТОГО репозитория
diagram_filename = diagram_filename_for_repo(repo)  # "hadoop.json"
graph_data = load_diagram("data/diagrams_normalized/hadoop.json")
# graph_data = {
#   "nodes": [3234 classes],
#   "edges": [11934 relationships]
# }
```

### Шаг 2: Запуск MCP-сервера (один сервер = один граф)

```python
# Создаём временный файл с графом
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
    json.dump(graph_data, tmp)  # Записываем весь граф hadoop
    graph_file = tmp.name       # /tmp/tmp8x3kz1p.json

# Запускаем MCP-сервер как subprocess
server_params = StdioServerParameters(
    command=sys.executable,
    args=["src/approaches/human_like_agent/mcp_server.py", graph_file],
)

# MCP-сервер загружает граф В ПАМЯТЬ:
# - Создаёт nodes_by_id = {node_id: node_data}
# - Создаёт adjacency lists: outgoing, incoming
# - Граф ОСТАЁТСЯ в памяти сервера до его завершения
```

### Шаг 3: Агент исследует граф

```
┌──────────────────────┐         stdio          ┌──────────────────────┐
│   LLM Agent          │ ←──────────────────→   │   MCP Server         │
│   (OpenAI)           │                        │   (subprocess)       │
│                      │                        │                      │
│  System Prompt +     │                        │  Graph in memory:    │
│  User Query +        │                        │  • nodes_by_id       │
│  Anchors             │                        │  • outgoing edges    │
│                      │                        │  • incoming edges    │
└──────────────────────┘                        └──────────────────────┘
         │                                                  │
         │  1. get_node_details("FileSystem")             │
         │ ──────────────────────────────────────────────→ │
         │                                                  │
         │  ← Returns: {type, methods, params, ...}        │
         │ ←────────────────────────────────────────────── │
         │                                                  │
         │  2. get_neighbors("FileSystem")                 │
         │ ──────────────────────────────────────────────→ │
         │                                                  │
         │  ← Returns: {outgoing: [...], incoming: [...]}  │
         │ ←────────────────────────────────────────────── │
         │                                                  │
         │  3. get_node_details("Path")                    │
         │ ──────────────────────────────────────────────→ │
         │                                                  │
         │  ... (up to max_steps tool calls)               │
         │                                                  │
         │  Final: {"required": [...], "useful": [...]}    │
         │ ──────────────────────────────────────────────→ │
         │                                                  │
```

### Шаг 4: Завершение и очистка

```python
# context manager автоматически:
# 1. Завершает MCP-сервер (subprocess.terminate())
# 2. Закрывает stdio streams
# 3. Удаляет временный файл

finally:
    os.unlink(graph_file)  # Удаляем /tmp/tmp8x3kz1p.json
```

## Важные детали

### 1. Граф загружается ОДИН РАЗ на сэмпл

```python
# MCP-сервер (в __init__):
self.graph_data = graph_data  # ВСЯ диаграмма в памяти
self.nodes_by_id = {node["node_id"]: node for node in graph_data["nodes"]}

# Adjacency lists для быстрого поиска соседей
self.outgoing = {}  # {node_id: [(target, edge_type), ...]}
self.incoming = {}  # {node_id: [(source, edge_type), ...]}
```

**Размер в памяти:**
- Activiti: 3234 nodes, 11934 edges ≈ 40MB JSON
- Все данные в RAM сервера

### 2. Изоляция между сэмплами

```
Sample 1: apache/hadoop + query1
  ↓
  [MCP Server #1] → hadoop.json в памяти
  ↓
  Agent explores hadoop graph
  ↓
  [MCP Server #1 terminates]
  ↓ (граф очищается из памяти)

Sample 2: apache/flink + query2
  ↓
  [MCP Server #2] → flink.json в памяти
  ↓
  Agent explores flink graph
  ↓
  [MCP Server #2 terminates]
```

**Нет переиспользования:** Каждый сэмпл = новый сервер = новый граф

### 3. Коммуникация через stdio

```
┌─────────────────┐                    ┌─────────────────┐
│  Agent Process  │                    │  MCP Server     │
│  (main thread)  │                    │  (subprocess)   │
└─────────────────┘                    └─────────────────┘
        │                                      │
        │ stdin  ──────────────────────────→  │
        │        JSON-RPC request:             │
        │        {                             │
        │          "method": "call_tool",      │
        │          "params": {                 │
        │            "name": "get_neighbors",  │
        │            "args": {"node_id": "A"}  │
        │          }                           │
        │        }                             │
        │                                      │
        │ stdout ←──────────────────────────  │
        │        JSON-RPC response:            │
        │        {                             │
        │          "result": {                 │
        │            "outgoing": [...],        │
        │            "incoming": [...]         │
        │          }                           │
        │        }                             │
```

## Альтернативные архитектуры (НЕ реализовано)

### Вариант A: Долгоживущий сервер (НЕ используется)

```
┌──────────────────────────┐
│  MCP Server (daemon)     │
│  • load_graph(repo)      │  ← Можно менять граф динамически
│  • unload_graph(repo)    │
└──────────────────────────┘
         ↑
         │ HTTP/gRPC
         │
┌──────────────────────────┐
│  Runner                  │
│  1. server.load("hadoop")│
│  2. agent explores       │
│  3. server.load("flink") │
│  4. agent explores       │
└──────────────────────────┘
```

**Плюсы:** Меньше overhead на запуск сервера  
**Минусы:** Сложность, управление состоянием, нужен дополнительный tool `load_graph`

### Вариант B: In-process tools (БЕЗ MCP)

```python
# Не используем MCP, просто Python-функции
def get_node_details(graph, node_id):
    return graph.nodes_by_id[node_id]

# LLM function calling напрямую вызывает Python
tools = [get_node_details, get_neighbors, ...]
```

**Плюсы:** Проще, быстрее  
**Минусы:** Не соответствует требованию "настоящий MCP-сервер"

## Текущая реализация: Обоснование

**Почему один сервер на сэмпл?**

1. ✅ **Изоляция:** Нет утечек данных между сэмплами
2. ✅ **Простота:** Не нужно управлять состоянием (load/unload графов)
3. ✅ **MCP стандарт:** Соответствует концепции MCP (один сервер = один контекст)
4. ✅ **Надёжность:** Сбой на одном сэмпле не ломает весь процесс
5. ✅ **Параллелизм:** Можно легко распараллелить (каждый сэмпл = независимый процесс)

**Overhead:**
- Запуск subprocess: ~100-200ms
- Загрузка JSON в память: ~50-100ms для больших графов
- **Итого:** ~200-300ms на сэмпл (пренебрежимо мало по сравнению с LLM вызовами)

## Итого

### Твоё понимание ВЕРНОЕ:
✅ Один MCP-сервер = один граф (репозиторий)  
✅ Новый сэмпл = новый сервер  
✅ Граф загружается в память сервера один раз  

### Что исправлено:
✅ `sys.argv` → `argparse` в `mcp_server.py`

### Архитектура:
```
Sample (repo, query, sample_id)
  → Load anchors (stage2_anchors)
  → Load graph (diagrams_normalized)
  → Start MCP server (subprocess, temp file)
  → Agent explores via tools (stdio JSON-RPC)
  → Return {required, useful}
  → Terminate server, cleanup temp file
```

Всё ясно? Есть ещё вопросы? 🚀
