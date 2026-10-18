"""Launch Streamlit from the project command."""
import subprocess
import sys

from .service import ROOT


def main():
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), *sys.argv[1:]],
        cwd=ROOT,
    ))
