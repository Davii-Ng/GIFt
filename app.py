"""Compatibility entry point for the upstream Streamlit app."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "apps" / "streamlit_app.py"), run_name="__main__")
