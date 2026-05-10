# Как запустить бейзлайны

Этот документ описывает процесс запуска бейзлайнов (reference approaches) на вашем датасете.

## Обзор бейзлайнов

### Стандартные бейзлайны (query-agnostic + lexical)

1. **empty** — возвращает пустой граф
   - Recall = 0, Precision = 0
   - Baseline "floor" — минимальная точка отсчета

2. **full_diagram** — возвращает весь граф
   - Recall = 1.0, Precision очень низкая
   - Показывает F1 без какой-либо фильтрации

3. **random_subset** — случайный набор узлов
   - Выбирает N узлов случайным образом
   - Воспроизводимо по sample_id
   - Конфигурация: `size`, `seed`

4. **top_degree** — узлы с наибольшей степенью
   - Берет N узлов с максимальным количеством связей
   - Query-agnostic graph centrality baseline
   - Конфигурация: `size`

5. **bm25** — BM25 ранжирование
   - Классический sparse retrieval
   - Использует ту же сериализацию узлов, что и embedding retriever
   - Прямое сравнение: sparse vs dense retrieval
   - Требует: `pip install rank-bm25`
   - Конфигурация: `size`

### Oracle бейзлайны (видят ground truth)

⚠️ **Важно:** Oracle бейзлайны НЕ являются настоящими подходами — они видят `central_node` и/или аннотации, которые production pipeline никогда не должен видеть. Они дают верхние границы (upper bounds) и санity checks.

1. **oracle_central_plus_neighbors** — центральный узел + соседи 1-го уровня
   - Проверяет: достаточно ли "1 hop от правильного anchor"?
   - Если F1 ≈ 0.55, потолок для "anchor + 1-hop" около этого значения
   - Если F1 ≈ 0.30, золотой ответ не в 1-hop neighbourhood

2. **oracle_gold_only** — точный золотой ответ
   - Должен всегда давать F1 = 1.0
   - Если нет — ошибка в evaluator или в формате выходных данных

## Установка зависимостей

```bash
# Основные зависимости уже установлены из requirements.txt
# Дополнительно для BM25:
pip install rank-bm25
```

## Конфигурация

Настройки бейзлайнов в `configs/config.yaml`:

```yaml
approaches:
  random_subset:
    size: 5              # Количество узлов
    seed: 42             # Базовый seed

  top_degree:
    size: 5              # Количество top-degree узлов

  bm25:
    size: 5              # Количество top-BM25 узлов
```

## Запуск

### Вариант 1: Все бейзлайны одной командой (рекомендуется)

```bash
# Запустить все бейзлайны и получить сводный отчет
python scripts/run_all_baselines.py

# С ограничением на количество сэмплов
python scripts/run_all_baselines.py --limit 10

# Только для конкретного репозитория
python scripts/run_all_baselines.py --repo apache/hadoop

# Пропустить BM25 (если rank_bm25 не установлен)
python scripts/run_all_baselines.py --skip-bm25

# Пропустить oracle бейзлайны
python scripts/run_all_baselines.py --skip-oracle

# Только query-agnostic бейзлайны
python scripts/run_all_baselines.py --skip-bm25 --skip-oracle
```

### Вариант 2: Запуск отдельных бейзлайнов

```bash
# Один конкретный бейзлайн
python scripts/run.py --approach empty
python scripts/run.py --approach full_diagram
python scripts/run.py --approach random_subset
python scripts/run.py --approach top_degree
python scripts/run.py --approach bm25

# С параметрами
python scripts/run.py --approach bm25 --limit 5 --repo apache/hadoop

# Oracle бейзлайны (отдельный скрипт)
python scripts/run_oracle_baselines.py

# Конкретные oracle бейзлайны
python scripts/run_oracle_baselines.py --baselines central_plus_neighbors
python scripts/run_oracle_baselines.py --baselines gold_only
```

## Результаты

### Структура выходных файлов

```
data/results/
├── empty/
│   ├── sample_0001.json
│   ├── sample_0002.json
│   └── evaluation_report.json
├── full_diagram/
│   └── ...
├── random_subset/
│   └── ...
├── top_degree/
│   └── ...
├── bm25/
│   └── ...
├── oracle_central_plus_neighbors/
│   └── ...
├── oracle_gold_only/
│   └── ...
└── baselines_summary.json      # Сводный отчет (если использован run_all_baselines.py)
```

### Формат evaluation_report.json

```json
{
  "summary": {
    "precision": 0.123,
    "recall_required": 0.456,
    "recall_useful": 0.234,
    "recall_overall": 0.345,
    "f1_score": 0.178,
    "total_samples": 50,
    "total_nodes_predicted": 250,
    "total_nodes_gold": 500
  },
  "per_sample": [...]
}
```

### Формат baselines_summary.json

```json
{
  "timestamp": "2026-05-10 15:30:00",
  "dataset": "data/dataset.csv",
  "diagrams_dir": "data/diagrams_normalized",
  "filters": {
    "repo": "all",
    "limit": "all"
  },
  "baselines": [
    {
      "baseline": "empty",
      "success": true,
      "elapsed_s": 1.5,
      "evaluation": {
        "f1_score": 0.0,
        "precision": 0.0,
        "recall_overall": 0.0
      }
    },
    ...
  ]
}
```

## Интерпретация результатов

### Для query-agnostic бейзлайнов:

- **empty**: F1 должен быть 0 (санity check)
- **full_diagram**: Recall = 1.0, но очень низкая Precision
  - F1 показывает "baseline без фильтрации"
- **random_subset**: Если ваш подход дает F1 близкий к random, это плохо
  - Пример: random_subset F1=0.15, ваш подход F1=0.18 — подход плохой
- **top_degree**: Query-agnostic структурный baseline
  - Если F1 высокий — центральные узлы графа часто релевантны

### Для BM25:

- **Сравнение с embedding retriever**:
  - Если BM25 F1 ≈ dense retrieval F1 → embeddings не добавляют ценности
  - Если BM25 F1 < dense retrieval F1 → embeddings работают

### Для oracle бейзлайнов:

- **central_plus_neighbors**:
  - F1 = 0.55 → потолок для "anchor + 1-hop" ≈ 0.55
  - F1 = 0.30 → нужна другая стратегия обхода графа (не 1-hop)
  
- **gold_only**:
  - F1 должен быть 1.0
  - Если F1 ≠ 1.0 → ошибка в evaluator или формате данных

## Сравнение с основными подходами

После запуска бейзлайнов и основных подходов, используйте `scripts/ablation.py`:

```bash
python scripts/ablation.py \
    --approaches empty full_diagram random_subset top_degree bm25 \
                 anchor_neighbors rag_classes_filter \
    --output reports/$(date +%Y-%m-%d).ablation.json
```

Это создаст side-by-side таблицу для сравнения.

## Troubleshooting

### rank_bm25 не установлен

```bash
pip install rank-bm25
# или
python scripts/run_all_baselines.py --skip-bm25
```

### Диаграммы не найдены

Убедитесь, что диаграммы нормализованы:

```bash
python scripts/normalize_diagrams.py
```

### dataset.csv не найден

Создайте датасет:

```bash
python scripts/build_dataset.py
```

## См. также

- `src/approaches/baselines/README.md` — подробная документация по бейзлайнам
- `scripts/run.py --help` — справка по запуску подходов
- `scripts/ablation.py --help` — справка по сравнительному анализу
