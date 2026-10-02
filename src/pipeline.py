"""
Single orchestration entry point that wires the whole InsightForge system
together: load data -> compute analytics -> build knowledge base -> build
retriever/RAG chain. Both `app/streamlit_app.py` and the test suite import
this so there is exactly one way the pieces get assembled.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.analysis.summary_stats import build_full_analysis
from src.config import Settings, get_settings
from src.data.loader import load_dataset
from src.rag.chain import InsightForgeRAGChain
from src.rag.fact_documents import FactDocument, build_fact_documents


@dataclass
class InsightForgeApp:
    """Bundles the fully-built state of the assistant for a given dataset."""

    raw_data: pd.DataFrame
    analysis: dict[str, Any]
    fact_documents: list[FactDocument]
    chain: InsightForgeRAGChain
    settings: Settings


def build_app(data_path: Optional[Path] = None, settings: Optional[Settings] = None) -> InsightForgeApp:
    """Build the full InsightForge stack from a CSV path (or the configured default)."""
    settings = settings or get_settings()
    df = load_dataset(data_path)
    analysis = build_full_analysis(df)
    docs = build_fact_documents(analysis)
    chain = InsightForgeRAGChain(documents=docs, settings=settings)
    return InsightForgeApp(
        raw_data=df, analysis=analysis, fact_documents=docs, chain=chain, settings=settings,
    )
