"""Framework-independent inference and capture storage."""
import base64
import binascii
import io
import hashlib
import json
import logging
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from safetensors.torch import load_file

from training.model import DigitCNN
from workers.vision import VisionProcessor, binarize, guide_box

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)
INFERENCE_LOCK = threading.Lock()
MAX_IMAGE_BYTES = 8 * 1024 * 1024
ASSETS = json.loads((ROOT / "deployment-assets.json").read_text(encoding="utf-8"))
SETTING_KEYS = ("MODEL_PATH", "HF_MODEL_REPO", "HF_MODEL_FILENAME", "HF_MODEL_REVISION",
                "MODEL_SHA256", "HF_TOKEN", "CAPTURE_DATABASE_PATH")


class AssetError(RuntimeError):
    """An unavailable or incompatible deployed model asset."""


def apply_settings(settings):
    for key in SETTING_KEYS:
        if key in settings and key not in os.environ:
            os.environ[key] = str(settings[key])


def decode_frame(payload: bytes | str) -> np.ndarray:
    """Accept uploaded bytes or legacy data URLs; orient and bound browser images."""
    try:
        if isinstance(payload, bytes):
            raw = payload
        elif isinstance(payload, str) and payload.startswith(("data:image/jpeg;base64,", "data:image/png;base64,")):
            if len(payload) > MAX_IMAGE_BYTES * 4 // 3 + 100:
                raise ValueError("Image is too large. Choose an image smaller than 8 MB.")
            raw = base64.b64decode(payload.split(",", 1)[1], validate=True)
        else:
            raise ValueError("Choose a JPEG or PNG image.")
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("Image is too large. Choose an image smaller than 8 MB.")
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in ("JPEG", "PNG"):
                raise ValueError("Choose a JPEG or PNG image.")
            if min(image.size) < 32 or max(image.size) > 4096:
                raise ValueError("Image dimensions must be between 32 and 4096 pixels.")
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((1280, 1280))
            rgb = np.asarray(image)
    except (binascii.Error, UnidentifiedImageError, OSError) as exc:
        raise ValueError("Could not read this image. Choose a JPEG or PNG.") from exc
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def detection_box(frame):
    x0, y0, size = guide_box(frame.shape)
    gray = cv2.cvtColor(frame[y0:y0 + size, x0:x0 + size], cv2.COLOR_BGR2GRAY)
    binary = binarize(cv2.GaussianBlur(gray, (5, 5), 0))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    height, width = frame.shape[:2]
    return [(x0 + x) / width, (y0 + y) / height, w / width, h / height]


def preview_image(payload):
    frame = decode_frame(payload)
    x, y, size = guide_box(frame.shape)
    guide = frame.copy()
    cv2.rectangle(guide, (x, y), (x + size - 1, y + size - 1), (93, 201, 244), 3)
    box = detection_box(frame)
    if box:
        height, width = frame.shape[:2]
        bx, by, bw, bh = box
        cv2.rectangle(guide, (round(bx * width), round(by * height)),
                      (round((bx + bw) * width), round((by + bh) * height)), (255, 180, 80), 2)
    return cv2.cvtColor(guide, cv2.COLOR_BGR2RGB)


def weights_path():
    local = os.environ.get("MODEL_PATH")
    if local:
        path = Path(local)
        if not path.is_file():
            raise FileNotFoundError("Model weights are missing at MODEL_PATH. Fix the path or remove this setting to download from Hugging Face.")
        return path
    from huggingface_hub import hf_hub_download

    model = ASSETS["model"]
    repo = os.environ.get("HF_MODEL_REPO", model["repo_id"])
    filename = os.environ.get("HF_MODEL_FILENAME", model["filename"])
    revision = os.environ.get("HF_MODEL_REVISION", model["revision"])
    try:
        path = Path(hf_hub_download(repo_id=repo, filename=filename, revision=revision,
                                     token=os.environ.get("HF_TOKEN") or None))
    except Exception as exc:
        LOGGER.exception("Model download failed for %s", repo)
        raise AssetError(f"Could not download the model from {repo}. Check the repository, revision, and network connection; private models also need HF_TOKEN in Streamlit secrets. Then try again.") from exc
    expected = os.environ.get("MODEL_SHA256")
    if expected is None and (repo, filename, revision) == (model["repo_id"], model["filename"], model["revision"]):
        expected = model["sha256"]
    if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise AssetError("The downloaded weights do not match the configured SHA-256. Check the pinned model revision and deployment-assets.json.")
    return path


@lru_cache(maxsize=1)
def load_model():
    torch.set_num_threads(min(2, os.cpu_count() or 2))
    config = json.loads((ROOT / "config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = config.get("input_size", 28)
    model = DigitCNN(
        input_channels=config["input_channels"],
        conv_channels=tuple(config["conv_channels"]),
        num_classes=config["num_classes"],
        mlp_hidden_size=config["mlp_hidden_size"],
        input_size=(size, size) if isinstance(size, int) else tuple(size),
    )
    model.load_state_dict(load_file(weights_path()))
    return model.to(device).eval(), device


def confidence_text(confidence):
    percent = confidence * 100
    text = ">99.9% confidence" if percent >= 99.9 else f"{percent:.1f}% confidence"
    if percent < 60:
        text += " (not sure, try again?)"
    return text


def animation_data(digit):
    path = ROOT / "static/animations" / f"classification_animation_{digit}.gif"
    return f"app/static/animations/{path.name}" if path.is_file() else None


def classify_frame(payload):
    """Each request owns its image and SQLite connection; model access is serialized."""
    try:
        frame = decode_frame(payload)
        captured_at = time.time_ns() // 1_000
        with INFERENCE_LOCK:
            model, device = load_model()
            database_path = os.environ.get("CAPTURE_DATABASE_PATH", str(ROOT / "data/captures.sqlite3"))
            processor = VisionProcessor(database_path)
            try:
                processed = processor.process((frame, captured_at))
                with torch.inference_mode():
                    probabilities = torch.softmax(model(processed.tensor.to(device)), dim=1)[0].cpu().numpy()
                digit = int(np.argmax(probabilities))
                confidence = float(probabilities[digit])
                processor.save_prediction(processed.capture_id, digit, confidence)
            finally:
                processor.close_database()
        return {
            "digit": digit,
            "confidence": confidence_text(confidence),
            "probabilities": probabilities.tolist(),
            "animation": animation_data(digit),
            "device": str(device),
            "preprocessing": {
                "grayscale": processed.grayscale,
                "binary": processed.binary,
                "model_input": processed.tensor[0, 0].cpu().numpy(),
            },
        }
    except (ValueError, FileNotFoundError, AssetError) as exc:
        return {"error": str(exc)}
    except Exception:
        LOGGER.exception("Classification failed")
        return {"error": "Could not classify this frame. Please try again or check the server logs."}


def app_status():
    # Configuration only: the first prediction downloads weights into the Hub cache.
    local = os.environ.get("MODEL_PATH")
    available = Path(local).is_file() if local else bool(os.environ.get("HF_MODEL_REPO", ASSETS["model"]["repo_id"]))
    return {
        "ready": available,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "message": "" if available else "Model weights are missing at MODEL_PATH. Remove the setting to use the configured Hugging Face model.",
    }
