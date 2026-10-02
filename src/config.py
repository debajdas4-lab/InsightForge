"""
InsightForge — Central configuration.

All environment-driven settings live here so the rest of the codebase never
touches `os.environ` directly. This keeps the system 12-factor-friendly and
easy to test (just monkeypatch `get_settings()`).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Strongly-typed application settings, loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider ----------------------------------------------------
    insightforge_llm_provider: Literal["openai", "anthropic", "none"] = Field(
        default="none",
        description="Which LLM backend to use. 'none' runs InsightForge in "
        "stats-only / offline mode (no generative answers, only computed "
        "analytics) which is useful for CI and demos without API keys.",
    )

    openai_api_key: Optional[str] = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    anthropic_api_key: Optional[str] = None
    anthropic_chat_model: str = "claude-sonnet-4-5"

    # --- LangSmith / observability ----------------------------------------
    langchain_tracing_v2: bool = False
    langchain_api_key: Optional[str] = None
    langchain_project: str = "insightforge"

    # --- Data / storage ----------------------------------------------------
    insightforge_data_path: str = "data/raw/sales_data.csv"
    insightforge_vectorstore_dir: str = "data/processed/chroma_db"
    insightforge_log_level: str = "INFO"

    @property
    def data_path(self) -> Path:
        p = Path(self.insightforge_data_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def vectorstore_dir(self) -> Path:
        p = Path(self.insightforge_vectorstore_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def llm_available(self) -> bool:
        """True if a real LLM provider is configured and its key is present."""
        if self.insightforge_llm_provider == "openai":
            return bool(self.openai_api_key)
        if self.insightforge_llm_provider == "anthropic":
            return bool(self.anthropic_api_key)
        return False


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton. Call `get_settings.cache_clear()` in tests."""
    return Settings()
