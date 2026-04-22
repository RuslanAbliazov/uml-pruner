"""LLM client wrapper around OpenAI with retries and parallelism."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI, BadRequestError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ContextOverflowError(Exception):
    """Raised when the API rejects a request because the prompt exceeds the
    context window. This is a deterministic error — we don't retry it.
    """

    def __init__(self, message: str, original: Exception | None = None):
        super().__init__(message)
        self.original = original


def _is_context_overflow(exc: BaseException) -> bool:
    """Detect OpenAI's context_length_exceeded signal.

    OpenAI returns HTTP 400 with error.code == 'context_length_exceeded' when
    the prompt is too long.
    """
    if not isinstance(exc, BadRequestError):
        return False
    # openai>=1.x: exc has .code and/or .body
    code = getattr(exc, "code", None)
    if code == "context_length_exceeded":
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") or {}
        if isinstance(err, dict) and err.get("code") == "context_length_exceeded":
            return True
    # Last-resort text match
    msg = str(exc).lower()
    return "context length" in msg or "maximum context" in msg


def _should_retry(exc: BaseException) -> bool:
    """Retry everything except ContextOverflowError."""
    if isinstance(exc, ContextOverflowError):
        return False
    if _is_context_overflow(exc):
        return False
    return True


@dataclass
class LLMResponse:
    """Wraps raw LLM response."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str


class LLMClient:
    """Async OpenAI client with retries and JSON-mode support."""

    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 90,
        retry_attempts: int = 3,
        retry_delay: int = 2,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.base_url = base_url or None  # normalize empty string to None

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "API key not set. Provide via LLMClient(api_key=...), "
                "config llm.api_key, or OPENAI_API_KEY environment variable."
            )

        # NOTE: `model` is intentionally NOT passed to AsyncOpenAI(). The
        # OpenAI SDK takes only connection-level settings here (api_key,
        # base_url, timeout, headers, proxies...). The model name is sent
        # per-request in chat.completions.create(model=self.model, ...).
        # This matches how every OpenAI-compatible provider works.
        client_kwargs: dict[str, Any] = {"api_key": key, "timeout": timeout}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = AsyncOpenAI(**client_kwargs)
        logger.info(
            "LLMClient initialised: model=%s base_url=%s",
            self.model,
            self.base_url or "<default openai>",
        )

        # Aggregate usage stats (useful for debugging, not exposed in output)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = True,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a single chat request with retries."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_format = {"type": "json_object"} if json_mode else None

        @retry(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=self.retry_delay, min=1, max=30),
            retry=retry_if_exception(_should_retry),
            reraise=True,
        )
        async def _do_request() -> Any:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format
            try:
                return await self._client.chat.completions.create(**kwargs)
            except BadRequestError as e:
                if _is_context_overflow(e):
                    raise ContextOverflowError(
                        f"Context window exceeded: {e}", original=e
                    ) from e
                raise

        try:
            completion = await _do_request()
        except ContextOverflowError:
            # Deterministic: caller (autosplit) will handle by splitting.
            raise
        except Exception:
            logger.exception("LLM call failed after retries")
            raise

        content = completion.choices[0].message.content or ""
        usage = completion.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
        )

    async def call_batch(
        self,
        prompts: list[tuple[str, str]],
        max_concurrency: int = 5,
        json_mode: bool = True,
        max_tokens: int | None = None,
    ) -> list[LLMResponse]:
        """Run many LLM calls concurrently with bounded parallelism."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded(system_prompt: str, user_prompt: str) -> LLMResponse:
            async with semaphore:
                return await self.call(
                    system_prompt,
                    user_prompt,
                    json_mode=json_mode,
                    max_tokens=max_tokens,
                )

        tasks = [_bounded(sp, up) for sp, up in prompts]
        return await asyncio.gather(*tasks)

    def usage_summary(self) -> dict[str, int]:
        """Get aggregate usage across calls."""
        return {
            "total_calls": self.total_calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
        }
