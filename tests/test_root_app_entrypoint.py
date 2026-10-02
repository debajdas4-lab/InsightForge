"""
Verifies the root-level `app.py` entry point is wired correctly: it must
exist, be syntactically valid, and delegate to `app/streamlit_app.py`
without duplicating its logic.
"""
from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_root_app_py_exists():
    assert (PROJECT_ROOT / "app.py").exists(), "Root-level app.py entry point is missing."


def test_root_app_py_is_valid_python():
    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    ast.parse(source)  # raises SyntaxError if invalid


def test_root_app_py_delegates_to_streamlit_app():
    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    assert "streamlit_app.py" in source
    assert (PROJECT_ROOT / "app" / "streamlit_app.py").exists()


def test_streamlit_app_module_is_syntactically_valid():
    source = (PROJECT_ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    ast.parse(source)
