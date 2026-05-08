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
* ``reranker.*``                         — модель кросс-энкодера (только если
                                           ``anchor_selector == "reranker"``)
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
class RerankerSettings:
    """Параметры кросс-энкодера для альтернативного stage 2.

    Заполняется ИЗ секции ``reranker:`` YAML и используется только когда
    ``PipelineSettings.anchor_selector == "reranker"``. В режиме
    ``"llm"`` поле ``AnchorNeighborsSettings.reranker`` равно ``None`` —
    секция YAML может вовсе отсутствовать.
    """
    model_name: str
    device: str
    batch_size: int
    max_seq_length: int | None  # None == «дефолт модели»


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


ANCHOR_SELECTORS = ("llm", "reranker")


@dataclass(frozen=True)
class PipelineSettings:
    """Гиперпараметры самого подхода ``anchor_neighbors``.

    ``max_subgraph_nodes`` — потолок размера подграфа (anchors+соседи),
    отправляемого в LLM-прун. ``0`` означает «без потолка»; YAML может
    задать ``-1`` или ``null`` — оба интерпретируем как 0.
    ``n_anchors`` — сколько якорей оставить после stage 2.
    ``1`` сохраняет историческое поведение; ``>1`` — multi-anchor режим.
    Должно выполняться ``1 <= n_anchors <= n_candidates``.
    ``anchor_selector`` — какой из двух движков использует stage 2:
    ``"llm"`` (LLM выбирает один из top-K) или ``"reranker"``
    (cross-encoder ранжирует и берёт top-N).
    ``prune_steps`` — список имён шагов для stage 4 (прунинга).
                      ["single"] — одношаговый (дефолт, обратная совместимость).
                      ["identify_core", "classify_neighbors"] — двухшаговый.
                      Каждое имя соответствует паре промптов:
                      prompts/prune_<step>_system.txt и prompts/prune_<step>_user.txt.
                      Специальное имя "single" использует prune_system.txt / prune_user.txt.
    ``outputs_dir``     — куда писать per-sample JSON и debug JSONL.
                          Имя селектора автоматически дописывается как
                          подпапка (``.../llm/``, ``.../reranker/``), чтобы
                          результаты двух прогонов не затирали друг друга.
    ``llm_traces_dir``  — куда писать последний request/response каждого
                          LLM-этапа (см. `llm_trace.py`). Тот же приём с
                          подпапкой по имени селектора.
    """
    n_candidates: int
    n_anchors: int
    max_subgraph_nodes: int
    anchor_selector: str
    prune_steps: list[str]
    outputs_dir: Path
    llm_traces_dir: Path


@dataclass(frozen=True)
class AnchorNeighborsSettings:
    """Полный набор настроек, нужных пайплайну.

    ``reranker`` равен ``None``, когда ``pipeline.anchor_selector == "llm"``
    — в этом режиме секция ``reranker:`` в YAML может отсутствовать.
    """
    retriever: RetrieverSettings
    llm: LLMSettings
    pipeline: PipelineSettings
    reranker: RerankerSettings | None = None


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


def _coerce_optional_positive(raw: Any, label: str) -> int | None:
    """Принять YAML-значение, где None / -1 == «дефолт» (None).

    Используется для ``reranker.max_seq_length``: positive int — как есть,
    null / отрицательное / 0 — None (отдать модели её собственный дефолт).
    """
    if raw is None:
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        raise ConfigError(
            f"{label}: ожидалось целое или null, получили {raw!r}"
        )
    return n if n > 0 else None


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

    # ---- anchor_selector ---------------------------------------------------
    # Дефолт `"llm"` сохраняет старое поведение: если ключа нет в YAML,
    # подход работает ровно как раньше.
    anchor_selector_raw = (
        own_section.get("anchor_selector") if hasattr(own_section, "get") else None
    ) or "llm"
    anchor_selector = str(anchor_selector_raw).strip().lower()
    if anchor_selector not in ANCHOR_SELECTORS:
        raise ConfigError(
            f"approaches.anchor_neighbors.anchor_selector: ожидалось одно из "
            f"{ANCHOR_SELECTORS}, получили {anchor_selector_raw!r}"
        )

    # ---- outputs_dir / llm_traces_dir -------------------------------------
    # Необязательные ключи (есть разумные дефолты, завязанные на имя подхода).
    # Дополнительно дописываем подпапку с именем селектора, чтобы прогоны
    # `llm` и `reranker` лежали бок о бок и не затирали друг друга.
    def _opt_path(key: str, default: str) -> Path:
        raw = own_section.get(key) if hasattr(own_section, "get") else None
        return Path(raw) if raw else Path(default)

    n_candidates = _required_int(
        own_section, "n_candidates", "approaches.anchor_neighbors"
    )
    # n_anchors — необязательный, дефолт 1 (исторический режим). Жёсткие
    # инварианты проверяем здесь, чтобы pipeline ничего не валидировал.
    n_anchors_raw = (
        own_section.get("n_anchors") if hasattr(own_section, "get") else None
    )
    if n_anchors_raw is None:
        n_anchors = 1
    else:
        try:
            n_anchors = int(n_anchors_raw)
        except (TypeError, ValueError):
            raise ConfigError(
                f"approaches.anchor_neighbors.n_anchors: ожидалось целое, "
                f"получили {n_anchors_raw!r}"
            )
    if n_anchors < 1:
        raise ConfigError(
            f"approaches.anchor_neighbors.n_anchors должен быть >= 1, "
            f"получили {n_anchors}"
        )
    if n_anchors > n_candidates:
        raise ConfigError(
            f"approaches.anchor_neighbors.n_anchors ({n_anchors}) не может "
            f"превышать n_candidates ({n_candidates})"
        )

    # ---- prune_steps ------------------------------------------------------
    # Список имён шагов для stage 4. Дефолт: ["single"] (одношаговый прунинг).
    prune_steps_raw = (
        own_section.get("prune_steps") if hasattr(own_section, "get") else None
    )
    if prune_steps_raw is None:
        prune_steps = ["single"]
    elif isinstance(prune_steps_raw, list):
        prune_steps = [str(s).strip() for s in prune_steps_raw if s]
        if not prune_steps:
            raise ConfigError(
                "approaches.anchor_neighbors.prune_steps не должен быть пустым списком"
            )
    else:
        raise ConfigError(
            f"approaches.anchor_neighbors.prune_steps должен быть списком, "
            f"получили {type(prune_steps_raw).__name__}"
        )

    pipeline = PipelineSettings(
        n_candidates=n_candidates,
        n_anchors=n_anchors,
        max_subgraph_nodes=_coerce_subgraph_cap(
            own_section.get("max_subgraph_nodes")
            if hasattr(own_section, "get")
            else None
        ),
        anchor_selector=anchor_selector,
        prune_steps=prune_steps,
        outputs_dir=_opt_path("outputs_dir", "data/results/anchor_neighbors")
        / anchor_selector,
        llm_traces_dir=_opt_path(
            "llm_traces_dir", "data/llm_traces/anchor_neighbors"
        )
        / anchor_selector,
    )

    # ---- reranker (опционально) -------------------------------------------
    reranker: RerankerSettings | None = None
    if anchor_selector == "reranker":
        rr_section = _section(cfg, "reranker")
        reranker = RerankerSettings(
            model_name=_required(rr_section, "model", "reranker"),
            device=_required(rr_section, "device", "reranker"),
            batch_size=_required_int(rr_section, "batch_size", "reranker"),
            max_seq_length=_coerce_optional_positive(
                rr_section.get("max_seq_length")
                if hasattr(rr_section, "get")
                else None,
                "reranker.max_seq_length",
            ),
        )

    return AnchorNeighborsSettings(
        retriever=retriever, llm=llm, pipeline=pipeline, reranker=reranker
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
