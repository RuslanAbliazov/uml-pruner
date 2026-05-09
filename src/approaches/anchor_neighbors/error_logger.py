"""Логирование ошибок этапов в понятные файлы для отладки.

При любой ошибке на любом этапе создаётся файл с полным traceback и контекстом.

Структура на диске:
    data/errors/anchor_neighbors/{anchor_selector}/
        stage1_retrieve/
            {sample_id}.error.txt  — полный traceback + входные данные
        stage2_anchor/
            {sample_id}.error.txt
        stage3_neighbors/
            {sample_id}.error.txt
        stage4_prune/
            {sample_id}.error.txt

Каждый файл содержит:
    - Полный traceback (как в консоли)
    - Входные параметры этапа
    - Время ошибки
    - Информацию для воспроизведения
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


class ErrorLogger:
    """Логирует ошибки этапов в отдельные файлы для простой отладки."""

    def __init__(self, root_dir: Path) -> None:
        """
        Args:
            root_dir: базовая папка для ошибок (например, data/errors/anchor_neighbors/llm)
        """
        self._root = root_dir

    def log_stage_error(
        self,
        stage_name: str,
        sample_id: str,
        exception: Exception,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Записать ошибку этапа в файл.

        Args:
            stage_name: название этапа (retrieve, anchor, neighbors, prune)
            sample_id: идентификатор сэмпла
            exception: перехваченное исключение
            context: дополнительный контекст (входные данные, параметры)
        """
        stage_dir = self._root / f"stage_{stage_name}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        error_file = stage_dir / f"{_safe_filename(sample_id)}.error.txt"

        # Полный traceback
        tb_lines = traceback.format_exception(
            type(exception), exception, exception.__traceback__
        )
        tb_str = "".join(tb_lines)

        # Формируем понятный файл
        content = []
        content.append("=" * 80)
        content.append(f"ERROR ON STAGE: {stage_name}")
        content.append("=" * 80)
        content.append(f"Sample ID: {sample_id}")
        content.append(f"Time: {datetime.now().isoformat()}")
        content.append(f"Error type: {type(exception).__name__}")
        content.append(f"Error message: {str(exception)}")
        content.append("")
        content.append("=" * 80)
        content.append("FULL TRACEBACK (without any fucking wrappers):")
        content.append("=" * 80)
        content.append(tb_str)

        if context:
            content.append("")
            content.append("=" * 80)
            content.append("CONTEXT (inputs, parameters):")
            content.append("=" * 80)
            content.append(json.dumps(context, ensure_ascii=False, indent=2, default=str))

        error_file.write_text("\n".join(content), encoding="utf-8")

        # Также выводим в stderr
        print(
            f"\n{'!' * 80}\n"
            f"ERROR LOGGED: {error_file}\n"
            f"Stage: {stage_name} | Sample: {sample_id}\n"
            f"{type(exception).__name__}: {str(exception)}\n"
            f"{'!' * 80}\n",
            file=__import__("sys").stderr,
        )


def _safe_filename(s: str) -> str:
    """Превратить sample_id в безопасное имя файла."""
    import re
    if not s:
        return "unknown"
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    return cleaned[:120]
