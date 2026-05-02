"""Запись «сырых» запросов/ответов LLM в плоские файлы для дебага.

Назначение — простое и узкое: после прогона хочется точно увидеть, что
именно мы отправили в модель и что она вернула. Без сторонних обёрток
типа OpenAI tracing, чтобы можно было открыть текстовый файл, прочитать,
скопировать в плейграунд.

Раскладка на диске:

    <root>/
      anchor/
        <sample_id>.req.txt    — system + user, отправленные на этап anchor
        <sample_id>.resp.txt   — то, что вернула LLM (raw content)
      prune/
        <sample_id>.req.txt
        <sample_id>.resp.txt

«Только последние»: каждый новый прогон по тому же `sample_id` ПЕРЕЗАПИСЫВАЕТ
оба файла. Так на диске всегда лежит последний выполненный запрос, и его
легко открыть «прямо сейчас».

Безопасность имён: `sample_id` приходит снаружи и может содержать что-то
неожиданное (слэши, двоеточия). Нормализуем в `_safe_name` — оставляем
только безопасный набор символов, остальные → `_`. Идентификаторы у нас
всегда либо UUID-подобные, либо `sample_NNNN`, поэтому коллизий не будет.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.approaches.anchor_neighbors.stage_outputs import StageName


_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_name(s: str) -> str:
    """Превратить произвольную строку в безопасное имя файла."""
    if not s:
        return "unknown"
    cleaned = _UNSAFE_RE.sub("_", s)
    # Подрежем длину: на случай, если sample_id вдруг стал длинным.
    return cleaned[:120]


class LLMTracer:
    """Сохраняет ровно один последний request/response на (stage, sample_id).

    Использование:

        tracer = LLMTracer(Path("data/llm_traces"))
        tracer.record_request(StageName.ANCHOR, sample_id, system, user)
        tracer.record_response(StageName.ANCHOR, sample_id, raw_content)
    """

    # Этапы, по которым мы вообще ходим в LLM. Этапы 1 и 3 — без LLM,
    # для них вызовы записи будут отказом (assert), чтобы случайно не
    # размазать трейсы по этапам, у которых их быть не должно.
    _LLM_STAGES = {StageName.ANCHOR, StageName.PRUNE}

    def __init__(self, root: Path) -> None:
        self._root = root

    # --- запись ---------------------------------------------------------

    def record_request(
        self,
        stage: StageName,
        sample_id: str,
        system_prompt: str,
        user_prompt: str,
    ) -> None:
        """Записать system+user в `<root>/<stage>/<sample_id>.req.txt`."""
        path = self._path(stage, sample_id, suffix="req")
        # Прозрачный двухсекционный формат с разделителем — удобно глазами.
        path.write_text(
            "===== SYSTEM =====\n"
            f"{system_prompt}\n"
            "===== USER =====\n"
            f"{user_prompt}\n",
            encoding="utf-8",
        )

    def record_response(
        self,
        stage: StageName,
        sample_id: str,
        raw_content: str,
    ) -> None:
        """Записать сырой контент ответа в `<root>/<stage>/<sample_id>.resp.txt`."""
        path = self._path(stage, sample_id, suffix="resp")
        path.write_text(raw_content or "", encoding="utf-8")

    def record_error(
        self,
        stage: StageName,
        sample_id: str,
        error_repr: str,
    ) -> None:
        """Если запрос упал ДО получения ответа, кладём текст ошибки в .resp.txt.

        Так файл `.resp.txt` всегда «парный» к `.req.txt`, и причина сбоя
        не теряется — это именно то, что хочется видеть последним.
        """
        path = self._path(stage, sample_id, suffix="resp")
        path.write_text(f"<<LLM CALL FAILED>>\n{error_repr}\n", encoding="utf-8")

    # --- внутренности --------------------------------------------------

    def _path(self, stage: StageName, sample_id: str, *, suffix: str) -> Path:
        assert stage in self._LLM_STAGES, (
            f"Этап {stage.value} не использует LLM — для него трейс не нужен"
        )
        folder = self._root / stage.value
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{_safe_name(sample_id)}.{suffix}.txt"
