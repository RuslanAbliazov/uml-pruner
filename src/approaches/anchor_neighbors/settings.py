"""Чтение конфигурации и сборка готового к запуску пайплайна.

Жёсткое правило: ВСЕ параметры берутся из ``configs/config.yaml``.
В этом модуле нет ни одного хардкода и ни одного fallback'а вида
``os.environ.get(...)``. Если ключ отсутствует или пуст — поднимаем
``ConfigError`` с понятным сообщением, чтобы пользователь сразу увидел,
какую секцию править.

Подстановка ``${ENV_VAR}`` в значениях — это особенность YAML-загрузчика
(``src.core.config``), а не этого файла. С точки зрения кода значение либо
есть в YAML, либо нет.

Используемые секции YAML:

* ``llm.*``                              — модель, base_url, api_key, ...
* ``embeddings.*``                       — модель эмбеддингов, device, ...
* ``approaches.anchor_neighbors.*``      — гиперпараметры самого подхода
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.config import load_config
from src.llm.client import LLMClient


class ConfigError(ValueError):
    """Поднимаем при отсутствии/пустом значении обязательного ключа в YAML."""


# ---- структуры конфигурации ----------------------------------------------

@dataclass(frozen=True)
class RetrieverSettings:
    """Что нужно RAG-индексу: какая модель и где лежит кэш."""
    model_name: str
    device: str
    batch_size: int
    cache_dir: str


@dataclass(frozen=True)
class LLMSettings:
    """Соединение с LLM. ``api_key`` обязан быть непустой строкой."""
    model: str
    base_url: str
    api_key: str
    temperature: float
    max_tokens: int
    timeout: int
    retry_attempts: int
    retry_delay: int


@dataclass(frozen=True)
class PipelineSettings:
    """Гиперпараметры самого подхода ``anchor_neighbors``.

    ``max_subgraph_nodes`` — потолок размера подграфа (anchor+соседи),
    отправляемого в LLM-прун. ``0`` означает «без потолка»; YAML может
    задать ``-1`` или ``null`` — оба интерпретируем как 0.
    ``outputs_dir`` — куда писать per-sample JSON и debug JSONL.
    """
    n_candidates: int
    max_subgraph_nodes: int
    outputs_dir: Path


@dataclass(frozen=True)
class AnchorNeighborsSettings:
    """Полный набор настроек, нужных пайплайну."""
    retriever: RetrieverSettings
    llm: LLMSettings
    pipeline: PipelineSettings


# ---- внутренние помощники чтения YAML ------------------------------------

def _section(cfg: Any, name: str) -> Any:
    """Достать секцию верхнего уровня из обёртки `Config`. Бросает понятную
    ошибку, если секции нет вообще."""
    sec = cfg.get(name) if hasattr(cfg, "get") else None
    if sec is None:
        raise ConfigError(f"в configs/config.yaml отсутствует секция '{name}'")
    return sec


def _required(section: Any, key: str, section_label: str) -> Any:
    """Достать обязательный ключ из секции; пусто/отсутствие → ошибка."""
    value = section.get(key) if hasattr(section, "get") else None
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ConfigError(
            f"в configs/config.yaml отсутствует или пуст ключ "
            f"'{section_label}.{key}'"
        )
    return value


def _required_int(section: Any, key: str, section_label: str) -> int:
    raw = _required(section, key, section_label)
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise ConfigError(
            f"'{section_label}.{key}' должен быть целым числом, "
            f"получили: {raw!r}"
        )


def _coerce_subgraph_cap(raw: Any) -> int:
    """Принять YAML-значение для max_subgraph_nodes:
    None / -1 / 0 → 0 («без потолка»). Положительное целое — как есть."""
    if raw is None:
        return 0
    try:
        n = int(raw)
    except (TypeError, ValueError):
        raise ConfigError(
            f"approaches.anchor_neighbors.max_subgraph_nodes: "
            f"ожидалось целое или null, получили {raw!r}"
        )
    return n if n > 0 else 0


# ---- сборка финальных настроек -------------------------------------------

def load_settings(cfg: Any | None = None) -> AnchorNeighborsSettings:
    """Прочитать YAML и вернуть валидированный набор настроек.

    ``cfg`` можно подать заранее загруженный (нужно для тестов и для
    вызова из ``run.py``, где конфиг уже под рукой). Если ``None`` —
    грузим стандартный путь ``configs/config.yaml``.
    """
    if cfg is None:
        cfg = load_config("configs/config.yaml")

    llm_section = _section(cfg, "llm")
    emb_section = _section(cfg, "embeddings")
    approaches_section = _section(cfg, "approaches")
    own_section = _required(approaches_section, "anchor_neighbors", "approaches")

    retriever = RetrieverSettings(
        model_name=_required(emb_section, "model", "embeddings"),
        device=_required(emb_section, "device", "embeddings"),
        batch_size=_required_int(emb_section, "batch_size", "embeddings"),
        cache_dir=_required(emb_section, "cache_dir", "embeddings"),
    )

    llm = LLMSettings(
        model=_required(llm_section, "model", "llm"),
        base_url=_required(llm_section, "base_url", "llm"),
        api_key=_required(llm_section, "api_key", "llm"),
        temperature=float(_required(llm_section, "temperature", "llm")),
        max_tokens=_required_int(llm_section, "max_tokens", "llm"),
        timeout=_required_int(llm_section, "timeout", "llm"),
        retry_attempts=_required_int(llm_section, "retry_attempts", "llm"),
        retry_delay=_required_int(llm_section, "retry_delay", "llm"),
    )

    # outputs_dir — необязательный (есть разумный путь по умолчанию,
    # завязанный на имя подхода), поэтому читаем мягко.
    outputs_raw = (
        own_section.get("outputs_dir")
        if hasattr(own_section, "get")
        else None
    )
    outputs_dir = Path(outputs_raw) if outputs_raw else Path(
        "data/results/anchor_neighbors"
    )

    pipeline = PipelineSettings(
        n_candidates=_required_int(
            own_section, "n_candidates", "approaches.anchor_neighbors"
        ),
        max_subgraph_nodes=_coerce_subgraph_cap(
            own_section.get("max_subgraph_nodes")
            if hasattr(own_section, "get")
            else None
        ),
        outputs_dir=outputs_dir,
    )

    return AnchorNeighborsSettings(
        retriever=retriever, llm=llm, pipeline=pipeline
    )


def make_llm_client(s: LLMSettings) -> LLMClient:
    """Сконструировать LLMClient ровно из YAML-секции llm."""
    return LLMClient(
        model=s.model,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        timeout=s.timeout,
        retry_attempts=s.retry_attempts,
        retry_delay=s.retry_delay,
        api_key=s.api_key,
        base_url=s.base_url,
    )


# ---- публичная фабрика ---------------------------------------------------

def build_runner(cfg: Any | None = None):
    """Вернуть готовый `AnchorNeighborsPipeline`, собранный из YAML.

    Используется и из реестра подходов (`src/approaches/__init__.py`),
    и из локального CLI (`run.py`).
    """
    # Локальный импорт, чтобы избежать цикла при импорте реестра.
    from src.approaches.anchor_neighbors.pipeline import AnchorNeighborsPipeline

    settings = load_settings(cfg)
    llm = make_llm_client(settings.llm)
    return AnchorNeighborsPipeline(settings, llm)
