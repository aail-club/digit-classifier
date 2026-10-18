"""Download and validate the production model before deploying to Community Cloud."""
import argparse
import os
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digit_classifier import service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    errors = []
    settings = ROOT / ".streamlit/secrets.toml"
    if settings.is_file():
        service.apply_settings(tomllib.loads(settings.read_text(encoding="utf-8")))
    from huggingface_hub import HfApi

    model = service.ASSETS["model"]
    try:
        info = HfApi().model_info(
            os.environ.get("HF_MODEL_REPO", model["repo_id"]),
            revision=os.environ.get("HF_MODEL_REVISION", model["revision"]), files_metadata=True,
        )
        print(f"Model repository is reachable at commit {info.sha}.")
        filename = os.environ.get("HF_MODEL_FILENAME", model["filename"])
        if not any(item.rfilename == filename for item in info.siblings):
            raise ValueError(f"Model file is missing: {filename}")
        # Ignore any local MODEL_PATH: verify the production download path.
        previous = os.environ.pop("MODEL_PATH", None)
        previous_database = os.environ.get("CAPTURE_DATABASE_PATH")
        try:
            service.load_model.cache_clear()
            service.load_model()
            with tempfile.TemporaryDirectory(prefix="digit-deploy-") as temporary:
                os.environ["CAPTURE_DATABASE_PATH"] = str(Path(temporary) / "captures.sqlite3")
                for digit in range(10):
                    result = service.classify_frame((ROOT / f"static/examples/digit_{digit}.png").read_bytes())
                    if result.get("digit") != digit:
                        raise ValueError(f"Example {digit} failed: {result.get('error', result.get('digit'))}")
        finally:
            if previous is not None:
                os.environ["MODEL_PATH"] = previous
            if previous_database is None:
                os.environ.pop("CAPTURE_DATABASE_PATH", None)
            else:
                os.environ["CAPTURE_DATABASE_PATH"] = previous_database
            service.load_model.cache_clear()
        print("Downloaded checkpoint passes SHA-256, architecture, and all ten example predictions.")
    except Exception as error:
        errors.append(f"MODEL: {error}")
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        raise SystemExit(1)
    print("Remote model deployment checks passed.")


if __name__ == "__main__":
    main()
