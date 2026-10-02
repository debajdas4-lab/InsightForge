from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generate_sample_data import generate_sales_data


@pytest.fixture(scope="session")
def sample_df():
    """A small, deterministic in-memory dataset shared across the test suite."""
    return generate_sales_data(n_rows=600, seed=7)
