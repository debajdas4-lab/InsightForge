"""
Lazily constructs a chat model for whichever provider is configured.

Kept separate from `chain.py` so both the RAG chain and the evaluation
module (`src.evaluation.evaluator`, which needs an LLM-as-judge) share one
source of truth for "how do we get a model instance".
"""
from __future__ import annotations

from src.config import Settings, get_settings


class LLMNotAvailableError(RuntimeError):
    """Raised when no usable LLM backend is configured/installed."""


def get_chat_model(settings: Settings | None = None, temperature: float = 0.2):
    """Return a LangChain-compatible chat model instance.

    Raises `LLMNotAvailableError` if no provider is configured or the
    required package isn't installed — callers (chain.py, evaluator.py)
    are expected to catch this and fall back to extractive/offline mode.
    """
    settings = settings or get_settings()

    if settings.insightforge_llm_provider == "openai":
        if not settings.openai_api_key:
            raise LLMNotAvailableError("OPENAI_API_KEY is not set.")
        try:
            from langchain_openai import ChatOpenAI  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover
            raise LLMNotAvailableError("langchain-openai is not installed.") from exc
        return ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key,
            temperature=temperature,
        )

    if settings.insightforge_llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise LLMNotAvailableError("ANTHROPIC_API_KEY is not set.")
        try:
            from langchain_anthropic import ChatAnthropic  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover
            raise LLMNotAvailableError("langchain-anthropic is not installed.") from exc
        return ChatAnthropic(
            model=settings.anthropic_chat_model,
            api_key=settings.anthropic_api_key,
            temperature=temperature,
        )

    raise LLMNotAvailableError(
        "No LLM provider configured (INSIGHTFORGE_LLM_PROVIDER='none'/unset). "
        "Running in stats-only / offline mode."
    )
