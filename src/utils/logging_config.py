"""Centralized logging configuration for InsightForge."""
from __future__ import annotations

import logging

from src.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.insightforge_log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
