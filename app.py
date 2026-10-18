"""Hugging Face Spaces entry point (also works with `python app.py`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from digit_classifier.app import build_app, main

demo = build_app()

if __name__ == "__main__":
    main(demo)
