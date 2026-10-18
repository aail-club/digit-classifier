"""Streamlit Community Cloud entry point: streamlit run app.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from digit_classifier.app import render_app

render_app()
