"""LLM behind a provider-agnostic interface.

The default impl talks the OpenAI chat-completions protocol. Pointing ``GR_OPENAI_BASE_URL``
at any OpenAI-compatible server (e.g. Project 2's local model) swaps the backend with zero
code change.
"""

from __future__ import annotations

from typing import Protocol

from .config import settings
from .log import get_logger

log = get_logger(__name__)


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenAICompatibleClient:
    def __init__(self) -> None:
        from openai import OpenAI

        kwargs: dict = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self._client = OpenAI(**kwargs)
        self._model = settings.llm_model
        log.info("llm_ready", model=self._model, base_url=settings.openai_base_url or "openai")

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


def get_llm() -> LLMClient:
    return OpenAICompatibleClient()
