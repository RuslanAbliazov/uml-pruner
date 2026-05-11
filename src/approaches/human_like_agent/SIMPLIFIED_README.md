# Human-Like Agent (Упрощённая версия)

Agent-based подход для анализа UML диаграмм через OpenCode + MCP инструменты.

## Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                      run.py (CLI)                           │
│  1. Загрузить anchors из data/stage2_anchors/              │
│  2. Скопировать граф в current_uml.json                     │
│  3. Сформировать промпт (query + anchors + инструкции)     │
│  4. Запустить: opencode run "<prompt>"                      │
│  5. Парсить JSON из stdout                                  │
│  6. Сохранить результат + trace                             │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ subprocess
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  OpenCode (управляет всем)                  │
│  • LLM вызовы (с function calling)                          │
│  • Управление MCP-сервером                                  │
│  • Tool calling loop                                        │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ stdio (JSON-RPC)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            MCP Server (mcp_server.py)                       │
│  • get_node_details(node_id)                                │
│  • get_neighbors(node_id, edge_type?)                       │
│  • get_edge_details(source_id, target_id)                   │
│  • search_nodes(pattern)                                    │
│                                                             │
│  Graph: current_uml.json (меняется для каждого sample)      │
└─────────────────────────────────────────────────────────────┘
```

## Файлы

```
src/approaches/human_like_agent/
├── run.py                   # CLI для запуска (единственный entry point)
├── evaluate.py              # Оценка метрик (отдельно от run.py)
├── mcp_server.py            # MCP-сервер с инструментами
├── opencode.json            # Конфигурация OpenCode + MCP
├── current_uml.json         # "Слот" для текущего графа (генерируется run.py)
├── prompts/
│   ├── agent_system.txt     # System prompt
│   └── agent_user.txt       # User prompt template
└── README.md
```

## Использование

### 1. Запуск на сэмплах

```bash
# Из корня проекта или из src/approaches/human_like_agent/
python src/approaches/human_like_agent/run.py --limit 1

# Конкретный репозиторий
python src/approaches/human_like_agent/run.py --repo apache/hadoop

# Конкретный sample_id
python src/approaches/human_like_agent/run.py --sample-id abc123
```

**Результаты:**
- `data/results/human_like_agent/{repo}__{sample_id}.json` — финальный результат
- `data/llm_traces/human_like_agent/{sample_id}.trace.txt` — полный trace OpenCode

### 2. Оценка метрик

```bash
python src/approaches/human_like_agent/evaluate.py

# С кастомными путями
python src/approaches/human_like_agent/evaluate.py \
    --results-dir data/results/human_like_agent \
    --dataset data/dataset.csv \
    --output reports/evaluation.json
```

**Вывод:**
```
SUMMARY METRICS:
Total samples:     5
Precision:         0.756
Recall Required:   0.623
Recall Useful:     0.534
Recall Overall:    0.578
F1 Score:          0.655
```

## Workflow

### Полный пайплайн

```bash
# 1. Нормализовать диаграммы (если нужно)
python scripts/normalize_diagrams.py

# 2. Создать датасет (если нужно)
python scripts/build_dataset.py

# 3. Сгенерировать anchors
python scripts/run.py --approach anchor_neighbors --limit 5

# 4. Запустить human_like_agent
python src/approaches/human_like_agent/run.py --limit 5

# 5. Оценить качество
python src/approaches/human_like_agent/evaluate.py
```

## Как это работает

### 1. Загрузка данных (БЕЗ утечки!)

```python
# run.py загружает ТОЛЬКО:
anchors = load_anchors(sample_id, repo)  # ✅ из data/stage2_anchors/
query = sample["query"]                   # ✅ пользовательский запрос

# НЕ загружает:
# ❌ central_node (ground truth)
# ❌ entity_annotations (правильные ответы)
```

### 2. Подготовка графа

```python
# Копировать нужный граф в "слот"
shutil.copy(
    "data/diagrams_normalized/hadoop.json",
    "src/approaches/human_like_agent/current_uml.json"
)

# MCP-сервер читает current_uml.json при старте
```

### 3. Формирование промпта

```python
prompt = f"""
{system_prompt}  # Инструкции + описание tools

USER QUERY: {query}
ANCHOR CLASSES: {anchors}

[Инструкции для агента]

CRITICAL: Output final JSON:
{{
  "required": [...],
  "useful": [...]
}}
"""
```

### 4. Запуск OpenCode

```python
result = subprocess.run(
    ["opencode", "run", prompt],
    cwd="src/approaches/human_like_agent",
    capture_output=True
)

output = result.stdout  # Полный вывод агента
```

### 5. Парсинг результата

```python
# Извлечь JSON из вывода (regex)
json_pattern = r'```json\s*(\{[^`]+\})\s*```'
match = re.search(json_pattern, output)

if match:
    result_json = json.loads(match.group(1))
    # {"required": [...], "useful": [...]}
```

### 6. Сохранение

```python
# Результат
with open(f"{repo}__{sample_id}.json", "w") as f:
    json.dump(result_json, f)

# Trace (для дебага)
with open(f"{sample_id}.trace.txt", "w") as f:
    f.write(output)  # Полный вывод OpenCode
```

## Отличия от старой версии

### Старая версия (сложная)

- ❌ `runner.py` — 324 строки, LLM backend вручную
- ❌ `settings.py` — загрузка конфигов для LLM
- ❌ `AsyncOpenAI` — прямые API вызовы
- ❌ `mcp.ClientSession` — управление MCP вручную
- ❌ Зависимости: `openai`, `mcp` (client side)

### Новая версия (простая)

- ✅ `run.py` — 200 строк, просто запускает OpenCode
- ✅ Никакого LLM кода — всё делает OpenCode
- ✅ Никаких asyncio/ClientSession — OpenCode управляет MCP
- ✅ Зависимости: только `mcp` (server side для mcp_server.py)

## Trace для дебага

Полный вывод OpenCode сохраняется в `data/llm_traces/human_like_agent/{sample_id}.trace.txt`:

```
Agent: I'll start by inspecting the anchor classes.

Tool: get_node_details("org.apache.hadoop.fs.FileSystem")
Result: {
  "type": "class",
  "name": "org.apache.hadoop.fs.FileSystem",
  "methods": ["open()", "create()", ...]
}

Agent: Now let me check the neighbors...

Tool: get_neighbors("org.apache.hadoop.fs.FileSystem")
Result: {
  "outgoing": [{"node_id": "org.apache.hadoop.fs.Path", "edge_type": "Association"}],
  ...
}

[... 30-40 tool calls ...]

Agent: Based on my exploration, here's my final answer:

```json
{
  "required": ["org.apache.hadoop.fs.FileSystem", "org.apache.hadoop.fs.Path"],
  "useful": ["org.apache.hadoop.conf.Configuration"]
}
```
```

Этот trace полезен для:
- Понимания стратегии агента
- Отладки ошибок парсинга
- Анализа того, какие классы агент исследовал

## Конфигурация OpenCode

Файл `opencode.json`:

```json
{
  "mcp": {
    "graph-navigator": {
      "type": "local",
      "command": ["python", "mcp_server.py", "--repo-path", "current_uml.json"],
      "enabled": true
    }
  }
}
```

**Важно:**
- `current_uml.json` — относительный путь от `src/approaches/human_like_agent/`
- OpenCode запускается из этой же директории
- Граф меняется для каждого sample (копируется из `data/diagrams_normalized/`)

## Troubleshooting

### "Anchors not found"

```bash
# Сначала сгенерируй anchors
python scripts/run.py --approach anchor_neighbors --limit 1
```

### "Diagram not found"

```bash
# Нормализуй диаграммы
python scripts/normalize_diagrams.py
```

### "Could not parse JSON"

Проверь trace:
```bash
cat data/llm_traces/human_like_agent/{sample_id}.trace.txt
```

Агент мог:
- Не вывести JSON (забыл инструкцию)
- Вывести невалидный JSON (синтаксическая ошибка)
- Превысить max_steps до финального ответа

### OpenCode не видит MCP tools

Проверь, что:
1. `opencode.json` в правильной директории
2. `current_uml.json` существует (run.py должен создать)
3. MCP-сервер запускается: `python mcp_server.py --repo-path current_uml.json`

## См. также

- `ARCHITECTURE.md` — детальное описание lifecycle
- `OPENCODE_SETUP.md` — настройка OpenCode для диалогового режима
- `mcp_server.py` — реализация MCP инструментов
