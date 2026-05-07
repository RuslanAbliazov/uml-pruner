import argparse
import json
import statistics
from pathlib import Path


def graph_tokens(path: Path, const: float = 4.0) -> int:
    """Приблизительное количество токенов в JSON-графе (символы / const)."""
    raw = path.read_text(encoding="utf-8")
    return len(raw) / const


def main():
    parser = argparse.ArgumentParser(
        description="Оценка веса графов этапа 3 в токенах"
    )
    parser.add_argument(
        "--directory",
        type=Path,
        nargs="?",
        default=Path("data/stage3_graphs"),
        help="папка с JSON-файлами графов (по умолчанию data/stage3_graphs)",
    )
    parser.add_argument(
        "--constant",
        type=float,
        default=4.0,
        help="символов на токен (по умолчанию 4)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="показать вес каждого файла",
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        parser.error(f"{args.directory} не является папкой")

    files = sorted(args.directory.glob("*.json"))
    if not files:
        print("JSON-файлы не найдены.")
        return

    tokens_per_graph = []
    for fpath in files:
        t = graph_tokens(fpath, args.constant)
        tokens_per_graph.append(t)
        if args.verbose:
            print(f"{fpath.name}: {t:.1f} токенов")

    mean = statistics.mean(tokens_per_graph)
    std = statistics.stdev(tokens_per_graph) if len(tokens_per_graph) > 1 else 0.0
    ci_low = mean - 2 * std
    ci_high = mean + 2 * std

    print(f"\nГрафов: {len(tokens_per_graph)}")
    print(f"Средний вес графа (mean): {mean:.1f} токенов")
    print(f"Стандартное отклонение (σ): {std:.1f}")
    print(f"95% доверительный интервал (±2σ): [{ci_low:.1f}, {ci_high:.1f}]")
    print(f"Минимум: {min(tokens_per_graph):.1f}, Максимум: {max(tokens_per_graph):.1f}")


if __name__ == "__main__":
    main()