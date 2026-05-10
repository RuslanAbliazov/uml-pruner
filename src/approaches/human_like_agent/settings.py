"""Configuration loading for human_like_agent approach.

Reads from configs/config.yaml section 'approaches.human_like_agent'.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.config import load_config
from src.llm.client import LLMClient


class ConfigError(ValueError):
    """Raised when required config key is missing or empty."""


@dataclass(frozen=True)
class LLMSettings:
    """LLM connection settings."""
    model: str
    base_url: str
    api_key: str
    temperature: float
    max_tokens: int
    timeout: int
    retry_attempts: int
    retry_delay: int


@dataclass(frozen=True)
class HumanLikeAgentSettings:
    """Settings for human_like_agent approach.
    
    max_steps: Maximum number of tool calls the agent can make (to control costs)
    outputs_dir: Where to write per-sample JSON results
    llm_traces_dir: Where to write LLM request/response traces
    """
    max_steps: int
    outputs_dir: Path
    llm_traces_dir: Path
    llm: LLMSettings


def _section(cfg: Any, name: str) -> Any:
    """Extract top-level section from config."""
    sec = cfg.get(name) if hasattr(cfg, "get") else None
    if sec is None:
        raise ConfigError(f"Missing section '{name}' in configs/config.yaml")
    return sec


def _required(section: Any, key: str, section_label: str) -> Any:
    """Extract required key from section."""
    value = section.get(key) if hasattr(section, "get") else None
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ConfigError(
            f"Missing or empty key '{section_label}.{key}' in configs/config.yaml"
        )
    return value


def _required_int(section: Any, key: str, section_label: str) -> int:
    """Extract required integer key from section."""
    raw = _required(section, key, section_label)
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise ConfigError(
            f"'{section_label}.{key}' must be integer, got: {raw!r}"
        )


def _optional_path(section: Any, key: str, default: str) -> Path:
    """Extract optional path, falling back to default."""
    raw = section.get(key) if hasattr(section, "get") else None
    return Path(raw) if raw else Path(default)


def load_settings(cfg: Any | None = None) -> HumanLikeAgentSettings:
    """Load and validate settings from YAML.
    
    Args:
        cfg: Pre-loaded config (or None to load from configs/config.yaml)
        
    Returns:
        Validated settings
    """
    if cfg is None:
        cfg = load_config("configs/config.yaml")

    llm_section = _section(cfg, "llm")
    approaches_section = _section(cfg, "approaches")
    own_section = _required(approaches_section, "human_like_agent", "approaches")

    llm = LLMSettings(
        model=_required(llm_section, "model", "llm"),
        base_url=_required(llm_section, "base_url", "llm"),
        api_key=_required(llm_section, "api_key", "llm"),
        temperature=float(_required(llm_section, "temperature", "llm")),
        max_tokens=_required_int(llm_section, "max_tokens", "llm"),
        timeout=_required(llm_section, "timeout", "llm"),
        retry_attempts=_required_int(llm_section, "retry_attempts", "llm"),
        retry_delay=_required_int(llm_section, "retry_delay", "llm"),
    )

    max_steps = _required_int(own_section, "max_steps", "approaches.human_like_agent")
    
    outputs_dir = _optional_path(
        own_section, "outputs_dir", "data/results/human_like_agent"
    )
    llm_traces_dir = _optional_path(
        own_section, "llm_traces_dir", "data/llm_traces/human_like_agent"
    )

    return HumanLikeAgentSettings(
        max_steps=max_steps,
        outputs_dir=outputs_dir,
        llm_traces_dir=llm_traces_dir,
        llm=llm,
    )


def make_llm_client(s: LLMSettings) -> LLMClient:
    """Construct LLMClient from LLMSettings."""
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


def build_runner(cfg: Any | None = None):
    """Build runner from YAML config.
    
    Used by approach registry (src/approaches/__init__.py).
    """
    # Import here to avoid circular dependency
    from src.approaches.human_like_agent.runner import HumanLikeAgentRunner

    settings = load_settings(cfg)
    llm = make_llm_client(settings.llm)
    return HumanLikeAgentRunner(settings, llm)
