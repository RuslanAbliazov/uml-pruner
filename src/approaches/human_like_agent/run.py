#!/usr/bin/env python3
"""CLI для запуска OpenCode агента с MCP инструментами для анализа UML диаграмм.

Использование:
    python run.py --limit 1
    python run.py --repo apache/hadoop
    python run.py --sample-id abc123
    
Результаты сохраняются в data/results/human_like_agent/
Trace сохраняется в data/llm_traces/human_like_agent/
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Определить пути относительно этого скрипта
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "human_like_agent_1"
TRACES_DIR = PROJECT_ROOT / "data" / "llm_traces" / "human_like_agent_1"
STAGE2_ANCHORS_DIR = PROJECT_ROOT / "data" / "stage2_anchors_hashed_1"
DIAGRAMS_DIR = PROJECT_ROOT / "data" / "diagrams_normalized"
CURRENT_UML = SCRIPT_DIR / "current_uml.json"


def load_dataset(path: Path) -> list[dict[str, str]]:
    """Загрузить dataset.csv.
    
    Args:
        path: Путь к dataset.csv
        
    Returns:
        Список записей (sample_id, repo, query, central_node, entity_annotations)
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset не найден: {path}\n"
            f"Создай его: python scripts/build_dataset.py"
        )
    
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_anchors(sample_id: str, repo: str) -> list[str]:
    """Загрузить anchor классы из stage2 (БЕЗ утечки ground truth!).
    
    Args:
        sample_id: ID сэмпла
        repo: Название репозитория
        
    Returns:
        Список anchor node_ids
    """
    anchor_file = STAGE2_ANCHORS_DIR / f"{repo.replace('/', '_')}__{sample_id}.json"
    print(anchor_file)
    if not anchor_file.exists():
        raise FileNotFoundError(
            f"Anchors не найдены: {anchor_file}\n"
            f"Сначала запусти: python scripts/run.py --approach anchor_neighbors --limit 1"
        )
    
    with anchor_file.open() as f:
        data = json.load(f)
    
    return data.get("anchors", [])


def prepare_graph(repo: str) -> None:
    """Скопировать граф репозитория в current_uml.json для MCP-сервера.
    
    Args:
        repo: Название репозитория (например, "apache/hadoop")
    """
    # Определить имя файла диаграммы
    diagram_name = repo.split("/")[-1] + ".json"
    source = DIAGRAMS_DIR / diagram_name
    
    if not source.exists():
        raise FileNotFoundError(
            f"Диаграмма не найдена: {source}\n"
            f"Нормализуй диаграммы: python scripts/normalize_diagrams.py"
        )
    
    # Скопировать в current_uml.json
    shutil.copy(source, CURRENT_UML)
    print(f"  Loaded graph: {diagram_name} → current_uml.json")


def build_prompt(query: str, anchors: list[str]) -> str:
    """Сформировать промпт из шаблонов."""
    prompts_dir = SCRIPT_DIR / "prompts"
    system_prompt = (prompts_dir / "agent_system.txt").read_text()
    user_template = (prompts_dir / "agent_user.txt").read_text()
    
    user_prompt = user_template.format(
        query=query,
        anchors=json.dumps(anchors, indent=2),
    )
    
    return f"{system_prompt}\n\n{user_prompt}"


FINAL_TOOL_NAME = "mark_final_statuses"


def _extract_tool_output(state_output: object) -> str | None:
    """Извлечь строку вывода MCP-инструмента из поля state.output.

    OpenCode сохраняет вывод MCP-инструмента как строку (TextContent от MCP-сервера).
    На некоторых версиях вывод может прийти как список объектов TextContent —
    обрабатываем оба варианта.
    """
    if isinstance(state_output, str):
        return state_output
    if isinstance(state_output, list):
        # Список TextContent-объектов: [{"type": "text", "text": "..."}]
        parts = []
        for item in state_output:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "".join(parts)
    return None


def parse_final_result_from_events(output: str) -> dict | None:
    """Извлечь результат mark_final_statuses из потока JSON-событий OpenCode.

    Ожидается вывод `opencode run ... --format json`: одно JSON-событие на строку.
    Ищется последнее завершённое событие tool_use с tool == mark_final_statuses
    (имя может иметь префикс MCP-сервера, например graph-navigator_mark_final_statuses).

    Args:
        output: stdout от `opencode run --format json`

    Returns:
        dict с ключами required/useful или None если событие не найдено
    """
    last_result: dict | None = None

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # Не каждая строка должна быть валидным JSON (на всякий случай)
            continue

        if event.get("type") != "tool_use":
            continue

        part = event.get("part") or {}
        tool_name = part.get("tool") or ""
        # Имя MCP-инструмента может быть с префиксом сервера, поэтому endswith
        if not tool_name.endswith(FINAL_TOOL_NAME):
            continue

        state = part.get("state") or {}
        if state.get("status") != "completed":
            continue

        tool_output_str = _extract_tool_output(state.get("output"))
        if not tool_output_str:
            continue

        try:
            tool_payload = json.loads(tool_output_str)
        except json.JSONDecodeError:
            continue

        if not isinstance(tool_payload, dict):
            continue

        # Игнорируем ошибки MCP-сервера (он возвращает {"error": ...})
        if "error" in tool_payload:
            continue

        required = tool_payload.get("required_node_ids")
        useful = tool_payload.get("useful_node_ids")
        if not isinstance(required, list) or not isinstance(useful, list):
            continue

        # Берём ПОСЛЕДНИЙ успешный вызов (агент мог переделать ответ)
        last_result = {"required": required, "useful": useful}

    return last_result


def run_opencode(prompt: str, sample_id: str) -> tuple[str, dict | None]:
    """Запустить OpenCode с промптом.

    Использует `--format json`: вместо парсинга текстового ответа агента
    напрямую перехватывается результат MCP-инструмента mark_final_statuses
    из потока структурированных событий.

    Args:
        prompt: Промпт для агента
        sample_id: ID сэмпла (для логирования)

    Returns:
        Кортеж (полный_вывод, распарсенный_json или None)
    """
    print(f"  Running OpenCode agent...")

    # Запустить OpenCode из директории подхода в json-режиме
    result = subprocess.run(
        ["opencode", "run", prompt, "--format", "json"],
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        timeout=600  # 10 минут timeout
    )

    output = result.stdout

    # Сохранить trace (NDJSON поток событий)
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    trace_file = TRACES_DIR / f"{sample_id}.trace.jsonl"
    trace_file.write_text(output)
    print(f"  Trace saved: {trace_file}")

    # Если были ошибки в stderr
    if result.stderr:
        stderr_file = TRACES_DIR / f"{sample_id}.stderr.txt"
        stderr_file.write_text(result.stderr)
        print(f"  Stderr saved: {stderr_file}")

    # Извлечь результат напрямую из вызова mark_final_statuses
    result_json = parse_final_result_from_events(output)

    if result_json:
        print(
            f"  Parsed result: {len(result_json.get('required', []))} required, "
            f"{len(result_json.get('useful', []))} useful"
        )
    else:
        print(f"  WARNING: mark_final_statuses tool call not found in events")

    return output, result_json


def save_result(sample_id: str, repo: str, query: str, result_json: dict) -> Path:
    """Сохранить результат в JSON.
    
    Args:
        sample_id: ID сэмпла
        repo: Репозиторий
        query: Запрос
        result_json: Результат с required/useful
        
    Returns:
        Путь к сохранённому файлу
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    output_file = RESULTS_DIR / f"{repo.replace('/', '_')}__{sample_id}.json"
    
    output_data = {
        "sample_id": sample_id,
        "repo": repo,
        "query": query,
        "required": result_json.get("required", []),
        "useful": result_json.get("useful", []),
    }
    
    with output_file.open("w") as f:
        json.dump(output_data, f, indent=2)
    
    return output_file


def parse_args() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    p = argparse.ArgumentParser(
        description="Запуск OpenCode агента для анализа UML диаграмм",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset.csv",
        help="Путь к dataset.csv",
    )
    p.add_argument(
        "--repo",
        default="",
        help="Фильтр по репозиторию (например, apache/hadoop)",
    )
    p.add_argument(
        "--sample-id",
        default="",
        help="Конкретный sample_id для обработки",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Ограничить количество сэмплов (0 = все)",
    )
    return p.parse_args()


def main():
    """Основная логика."""
    args = parse_args()
    
    print("="*60)
    print("Human-Like Agent: OpenCode + MCP")
    print("="*60)
    
    # 1. Загрузить датасет
    try:
        dataset = load_dataset(args.dataset)
        print(f"Loaded {len(dataset)} samples from dataset")
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 2. Применить фильтры
    if args.repo:
        dataset = [s for s in dataset if s["repo"] == args.repo]
        print(f"Filtered to repo={args.repo}: {len(dataset)} samples")
    
    if args.sample_id:
        dataset = [s for s in dataset if s["sample_id"] == args.sample_id]
        print(f"Filtered to sample_id={args.sample_id}: {len(dataset)} samples")
    
    if args.limit > 0:
        dataset = dataset[:args.limit]
        print(f"Limited to {args.limit} samples")
    
    if not dataset:
        print("ERROR: No samples to process after filtering!", file=sys.stderr)
        sys.exit(1)
    
    print("="*60)
    
    # 3. Обработать каждый сэмпл
    success_count = 0
    failed_samples = []
    
    for i, sample in enumerate(dataset, 1):
        sample_id = sample["sample_id"]
        repo = sample["repo"]
        query = sample["query"]
        
        print(f"\n[{i}/{len(dataset)}] Processing sample: {sample_id}")
        print(f"  Repo: {repo}")
        print(f"  Query: {query[:80]}...")
        
        try:
            # 3.1 Загрузить anchors (БЕЗ утечки central_node/annotations!)
            anchors = load_anchors(sample_id, repo)
            print(f"  Anchors: {len(anchors)} classes")
            
            # 3.2 Подготовить граф для MCP
            prepare_graph(repo)
            
            # 3.3 Сформировать промпт
            prompt = build_prompt(query, anchors)
            
            # 3.4 Запустить OpenCode
            output, result_json = run_opencode(prompt, sample_id)
            
            if result_json:
                # 3.5 Сохранить результат
                output_file = save_result(sample_id, repo, query, result_json)
                print(f"  ✓ Result saved: {output_file}")
                success_count += 1
            else:
                print(f"  ✗ FAILED: Could not parse JSON from output")
                failed_samples.append(sample_id)
        
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed_samples.append(sample_id)
            import traceback
            traceback.print_exc()
    
    # 4. Сводка
    print("\n" + "="*60)
    print(f"Processed: {len(dataset)} samples")
    print(f"Success: {success_count}")
    print(f"Failed: {len(failed_samples)}")
    
    if failed_samples:
        print(f"Failed samples: {', '.join(failed_samples)}")
    
    print(f"\nResults: {RESULTS_DIR}")
    print(f"Traces: {TRACES_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
