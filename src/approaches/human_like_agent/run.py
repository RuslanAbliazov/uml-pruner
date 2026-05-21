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
import re
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


def parse_json_from_output(output: str) -> dict | None:
    """Извлечь JSON из вывода OpenCode (ищет required/useful)."""
    patterns = [
        r'```json\s*(\{[^`]+\})\s*```',  # Markdown блок
        r'\{[^{}]*"required_node_ids"[^{}]*"useful_node_ids"[^{}]*\}',  # get_final_result format
        r'\{[^{}]*"required"[^{}]*"useful"[^{}]*\}',  # Simple format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1) if '```' in pattern else match.group(0))
                # Normalize: get_final_result returns required_node_ids/useful_node_ids
                if "required_node_ids" in data:
                    return {
                        "required": data["required_node_ids"],
                        "useful": data["useful_node_ids"]
                    }
                return data
            except (json.JSONDecodeError, IndexError):
                continue
    
    return None


def run_opencode(prompt: str, sample_id: str) -> tuple[str, dict | None]:
    """Запустить OpenCode с промптом.
    
    Args:
        prompt: Промпт для агента
        sample_id: ID сэмпла (для логирования)
        
    Returns:
        Кортеж (полный_вывод, распарсенный_json или None)
    """
    print(f"  Running OpenCode agent...")
    
    # Запустить OpenCode из директории подхода
    result = subprocess.run(
        ["opencode", "run", prompt],
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        timeout=600  # 10 минут timeout
    )
    
    output = result.stdout
    
    # Сохранить trace
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    trace_file = TRACES_DIR / f"{sample_id}.trace.txt"
    trace_file.write_text(output)
    print(f"  Trace saved: {trace_file}")
    
    # Если были ошибки в stderr
    if result.stderr:
        stderr_file = TRACES_DIR / f"{sample_id}.stderr.txt"
        stderr_file.write_text(result.stderr)
        print(f"  Stderr saved: {stderr_file}")
    
    # Парсить JSON из вывода
    result_json = parse_json_from_output(output)
    
    if result_json:
        print(f"  Parsed JSON: {len(result_json.get('required', []))} required, {len(result_json.get('useful', []))} useful")
    else:
        print(f"  WARNING: Could not parse JSON from output")
    
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
