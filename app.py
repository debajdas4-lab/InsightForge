"""
InsightForge — root-level entry point.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Suppress Streamlit's internal use_container_width/width deprecation
# notice. Confirmed (via source inspection of deprecation_util.py) that
# Streamlit's own plotly_chart/dataframe internals trigger this on
# themselves during their width-parameter migration; it is not caused by
# application code. Patched to a no-op so the terminal stays clean.
import streamlit.deprecation_util
streamlit.deprecation_util.show_deprecation_warning = lambda *args, **kwargs: None

_STREAMLIT_APP = PROJECT_ROOT / "app" / "streamlit_app.py"
runpy.run_path(str(_STREAMLIT_APP), run_name="__main__")
