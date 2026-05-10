# Быстрый старт: Запуск бейзлайнов

## TL;DR

```bash
# 1. Установить зависимость для BM25
pip install rank-bm25

# 2. Запустить все бейзлайны
python scripts/run_all_baselines.py

# 3. Результаты в data/results/
```

## Что произойдет

Скрипт запустит **7 бейзлайнов**:

### Стандартные (5 штук):
- `empty` — пустой граф (F1=0)
- `full_diagram` — весь граф (Recall=1.0)
- `random_subset` — 5 случайных узлов
- `top_degree` — 5 узлов с максимальной степенью
- `bm25` — 5 узлов по BM25 ранжированию

### Oracle (2 штуки, видят ground truth):
- `oracle_central_plus_neighbors` — центральный узел + соседи
- `oracle_gold_only` — точный золотой ответ (F1=1.0)

## Результаты

```
data/results/
├── empty/evaluation_report.json
├── full_diagram/evaluation_report.json
├── random_subset/evaluation_report.json
├── top_degree/evaluation_report.json
├── bm25/evaluation_report.json
├── oracle_central_plus_neighbors/evaluation_report.json
├── oracle_gold_only/evaluation_report.json
└── baselines_summary.json          # ← Сводная таблица
```

## Сводная таблица

После запуска увидите таблицу:

```
Baseline                       Success    F1         Prec       Rec        Time(s)   
--------------------------------------------------------------------------------
empty                          ✓          0.000      0.000      0.000      1.2       
full_diagram                   ✓          0.156      0.089      0.998      1.5       
random_subset                  ✓          0.145      0.123      0.178      2.1       
top_degree                     ✓          0.234      0.198      0.289      2.3       
bm25                           ✓          0.312      0.278      0.356      3.8       
oracle_central_plus_neighbors  ✓          0.567      0.489      0.678      1.9       
oracle_gold_only               ✓          1.000      1.000      1.000      1.1       
```

## Опции

```bash
# Только query-agnostic (без BM25 и oracle)
python scripts/run_all_baselines.py --skip-bm25 --skip-oracle

# Ограничить 10 сэмплами (для быстрого теста)
python scripts/run_all_baselines.py --limit 10

# Конкретный репозиторий
python scripts/run_all_baselines.py --repo apache/hadoop

# Перезаписать существующие результаты
python scripts/run_all_baselines.py --overwrite

# Подробный вывод
python scripts/run_all_baselines.py --verbose
```

## Если rank-bm25 не установлен

```bash
python scripts/run_all_baselines.py --skip-bm25
```

## Интерпретация

1. **gold_only должен показать F1=1.0** — если нет, проблема с evaluator
2. **Ваш подход должен бить random_subset** — иначе подход плохой
3. **BM25 vs dense retrieval** — если BM25 лучше, embeddings бесполезны
4. **central_plus_neighbors** — верхняя граница для "anchor + 1-hop"

## Далее

- Полная документация: `BASELINES_HOWTO.md`
- Запуск основных подходов: `python src/approaches/anchor_neighbors/run.py`
- Сравнение: `python scripts/ablation.py --approaches empty bm25 anchor_neighbors`
