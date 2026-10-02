"""
Knowledge base creation: load the sales dataset and organize it into a
structured format suitable for both statistical analysis and retrieval
(Part 1, Step 2 of the brief).

Design choice: rather than chunking raw CSV rows for a vector store (which
is a poor fit for numeric/tabular data), InsightForge builds a *structured
knowledge base* of pre-aggregated, human-readable "fact documents" — one per
(time period / product / region / segment) slice. These fact documents are
what actually gets embedded and retrieved, which is far more effective for
grounding an LLM in accurate statistics than stuffing raw rows into it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import get_settings

REQUIRED_COLUMNS = [
    "order_id", "order_date", "region", "country", "product_category",
    "product_name", "customer_id", "customer_segment", "age_group",
    "units_sold", "unit_price", "revenue", "unit_cost", "profit",
]


@dataclass
class KnowledgeBase:
    """A structured, in-memory representation of the business dataset."""

    raw: pd.DataFrame
    fact_documents: list[dict] = field(default_factory=list)

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.raw["order_date"].min(), self.raw["order_date"].max()

    def as_frame(self) -> pd.DataFrame:
        return self.raw


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load and lightly validate the business dataset.

    Per the brief, this step focuses on analysis/structuring, not cleaning —
    we still run a minimal schema/sanity check so downstream analysis never
    silently operates on malformed data.
    """
    settings = get_settings()
    csv_path = path or settings.data_path
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Run "
            "`python -m src.data.generate_sample_data` to create a sample, "
            "or point INSIGHTFORGE_DATA_PATH at your own extract."
        )

    df = pd.read_csv(csv_path, parse_dates=["order_date"])

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Minimal, non-destructive sanity checks (not data cleaning).
    numeric_cols = ["units_sold", "unit_price", "revenue", "unit_cost", "profit"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=numeric_cols)
    n_dropped = n_before - len(df)
    if n_dropped:
        import logging
        logging.getLogger(__name__).warning(
            "Dropped %d rows with non-numeric values in numeric columns.", n_dropped
        )

    df["order_date"] = pd.to_datetime(df["order_date"])
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.to_period("M").astype(str)
    df["quarter"] = df["order_date"].dt.to_period("Q").astype(str)
    return df.reset_index(drop=True)


def build_knowledge_base(df: Optional[pd.DataFrame] = None) -> KnowledgeBase:
    """Load (if needed) and wrap the dataset into a `KnowledgeBase`.

    The heavy lifting of turning it into retrievable "fact documents" lives
    in `src.analysis.summary_stats`, which is intentionally decoupled from
    I/O so it can be unit tested with in-memory frames.
    """
    frame = df if df is not None else load_dataset()
    return KnowledgeBase(raw=frame)
