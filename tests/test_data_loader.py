from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import REQUIRED_COLUMNS, build_knowledge_base, load_dataset


def test_load_dataset_from_explicit_path(tmp_path, sample_df):
    csv_path = tmp_path / "sample.csv"
    sample_df.to_csv(csv_path, index=False)

    loaded = load_dataset(csv_path)
    assert set(REQUIRED_COLUMNS).issubset(loaded.columns)
    assert pd.api.types.is_datetime64_any_dtype(loaded["order_date"])
    assert len(loaded) == len(sample_df)


def test_load_dataset_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_dataset(missing_path)


def test_load_dataset_missing_columns_raises(tmp_path):
    bad_path = tmp_path / "bad.csv"
    pd.DataFrame({"order_id": [1], "order_date": ["2024-01-01"]}).to_csv(bad_path, index=False)
    with pytest.raises(ValueError):
        load_dataset(bad_path)


def test_build_knowledge_base_wraps_frame(sample_df):
    kb = build_knowledge_base(sample_df)
    assert kb.as_frame() is sample_df
    start, end = kb.date_range
    assert start <= end
