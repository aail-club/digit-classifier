"""Image processing, storage, and Streamlit session regression coverage."""
import base64
import json
import hashlib
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from PIL import Image
from streamlit.testing.v1 import AppTest

from digit_classifier import service
from training.model import DigitCNN
from workers.vision import VisionProcessor, guide_box


@pytest.fixture
def frame():
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.putText(image, "7", (260, 320), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 14)
    return image


def encode(image):
    ok, png = cv2.imencode(".png", image)
    assert ok
    return "data:image/png;base64," + base64.b64encode(png).decode()


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    config = json.loads((service.ROOT / "config.json").read_text())
    torch.manual_seed(42)
    model = DigitCNN(conv_channels=tuple(config["conv_channels"]), mlp_hidden_size=config["mlp_hidden_size"])
    # A deterministic UI fixture, not the production model: always choose digit 7.
    with torch.no_grad():
        model.classifier[-1].weight.zero_()
        model.classifier[-1].bias.zero_()
        model.classifier[-1].bias[7] = 8
    path = tmp_path / "fixture.safetensors"
    save_file(model.state_dict(), path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    monkeypatch.setenv("CAPTURE_DATABASE_PATH", str(tmp_path / "captures.sqlite3"))
    service.load_model.cache_clear()
    yield tmp_path / "captures.sqlite3"
    service.load_model.cache_clear()


def test_preprocessing_and_storage_roundtrip(frame, tmp_path):
    path = tmp_path / "captures.sqlite3"
    processor = VisionProcessor(path)
    try:
        result = processor.process((frame, 123456789))
        assert guide_box(frame.shape) == (176, 96, 288)
        assert result.tensor.shape == (1, 1, 28, 28)
        assert result.tensor.dtype == torch.float32
        assert 0 <= result.tensor.min() <= result.tensor.max() <= 1
        assert result.tensor.count_nonzero() > 0
        processor.save_prediction(result.capture_id, 7, .91)
    finally:
        processor.close_database()
    with sqlite3.connect(path) as db:
        timestamp, width, height, bitmap, prediction, confidence = db.execute(
            "SELECT captured_at_us,width,height,bitmap,prediction,confidence FROM captures"
        ).fetchone()
    pixels = np.unpackbits(np.frombuffer(bitmap, np.uint8), bitorder="big")[:width * height]
    np.testing.assert_array_equal(pixels.reshape(height, width) * 255, result.binary)
    assert (timestamp, prediction, confidence) == (123456789, 7, .91)


def test_outside_guide_box_does_not_change_tensor(frame, tmp_path):
    changed = frame.copy()
    changed[:, :150] = 0
    changed[:, 490:] = 0
    processor = VisionProcessor(tmp_path / "captures.sqlite3")
    try:
        a = processor.process((frame, 1))
        b = processor.process((changed, 2))
        torch.testing.assert_close(a.tensor, b.tensor, rtol=0, atol=0)
    finally:
        processor.close_database()


def test_inference_and_concurrent_capture_isolation(frame, checkpoint):
    payload = encode(frame)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(service.classify_frame, [payload] * 3))
    for result in results:
        assert "error" not in result, result
        assert len(result["probabilities"]) == 10
        assert sum(result["probabilities"]) == pytest.approx(1)
        assert result["digit"] == int(np.argmax(result["probabilities"]))
    with sqlite3.connect(checkpoint) as db:
        rows = db.execute("SELECT id,prediction,confidence FROM captures").fetchall()
    assert len({row[0] for row in rows}) == 3
    assert all(row[1] == results[0]["digit"] for row in rows)


def test_failures_are_actionable(frame, checkpoint, monkeypatch, tmp_path):
    assert service.classify_frame("not an image")["error"] == "Choose a JPEG or PNG image."
    assert "No digit" in service.classify_frame(encode(np.full_like(frame, 255)))["error"]
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.safetensors"))
    monkeypatch.delenv("HF_MODEL_REPO", raising=False)
    service.load_model.cache_clear()
    assert not service.app_status()["ready"]
    assert "Model weights are missing" in service.classify_frame(encode(frame))["error"]


@pytest.mark.parametrize("value,expected", [(1., ">99.9% confidence"), (.9997, ">99.9% confidence"), (.75, "75.0% confidence"), (.4, "40.0% confidence (not sure, try again?)")])
def test_confidence_wording(value, expected):
    assert service.confidence_text(value) == expected


def test_inference_does_not_import_training_dataset():
    assert "training.dataset" not in sys.modules
    assert "PySide6" not in sys.modules


def test_bundled_checkpoint_classifies_digit(frame, tmp_path, monkeypatch):
    path = service.ROOT / "weights/model.safetensors"
    if not path.is_file():
        pytest.skip("Bundled trained checkpoint not available")
    monkeypatch.setenv("MODEL_PATH", str(path))
    monkeypatch.setenv("CAPTURE_DATABASE_PATH", str(tmp_path / "captures.sqlite3"))
    service.load_model.cache_clear()
    try:
        result = service.classify_frame(encode(frame))
        assert "error" not in result, result
        assert result["digit"] == 7
        assert result["probabilities"][7] > .99
    finally:
        service.load_model.cache_clear()


def test_local_prerendered_gifs_if_available():
    for digit in range(10):
        path = service.ROOT / "static/animations" / f"classification_animation_{digit}.gif"
        if not path.is_file():
            pytest.skip("Optional local rendering files are not deployed")
        with Image.open(path) as gif:
            assert gif.size == (1920, 1080)
            assert gif.is_animated
            assert gif.n_frames == 1566
            assert gif.info.get("loop") == 0
        with path.open("rb") as stream:
            stream.seek(-1, 2)
            assert stream.read(1) == b";"
        assert service.animation_data(digit) == f"app/static/animations/{path.name}"


def test_uploaded_bytes_and_phone_orientation(frame):
    import io
    ok, png = cv2.imencode(".png", frame)
    assert ok
    np.testing.assert_array_equal(service.decode_frame(png.tobytes()), frame)
    exif = Image.Exif()
    exif[274] = 6
    stream = io.BytesIO()
    Image.new("RGB", (80, 40), "white").save(stream, format="JPEG", exif=exif)
    assert service.decode_frame(stream.getvalue()).shape[:2] == (80, 40)
    assert "Could not read" in service.classify_frame(b"invalid png")["error"]


def test_streamlit_flow_does_not_duplicate_inference(checkpoint):
    database = checkpoint
    service.load_model.cache_clear()
    try:
        app = AppTest.from_file(str(service.ROOT / "app.py"), default_timeout=30).run()
        assert not app.exception
        assert app.session_state["result"] is None
        app.button(key="classify").click().run()
        assert not app.exception
        assert app.session_state["result"]["digit"] == 7
        app.run()
        with sqlite3.connect(database) as db:
            assert db.execute("SELECT COUNT(*) FROM captures").fetchone()[0] == 1
        app.button(key="try_again").click().run()
        assert app.session_state["result"] is None
        assert app.session_state["input_epoch"] == 1
        app.radio[0].set_value("Upload a photo").run()
        assert app.button(key="classify").disabled
        assert not app.exception
    finally:
        service.load_model.cache_clear()


def test_streamlit_recovers_from_missing_model(tmp_path, monkeypatch, checkpoint):
    checkpoint_path = os.environ["MODEL_PATH"]
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.safetensors"))
    monkeypatch.delenv("HF_MODEL_REPO", raising=False)
    service.load_model.cache_clear()
    try:
        app = AppTest.from_file(str(service.ROOT / "app.py"), default_timeout=30).run()
        app.button(key="classify").click().run()
        assert not app.exception
        assert app.session_state["result"] is None
        assert "Model weights are missing" in app.error[0].value
        monkeypatch.setenv("MODEL_PATH", checkpoint_path)
        app.button(key="classify").click().run()
        assert app.session_state["result"]["digit"] == 7
    finally:
        service.load_model.cache_clear()


def test_remote_model_cache_and_integrity(checkpoint, monkeypatch):
    import copy
    import huggingface_hub
    path = Path(os.environ["MODEL_PATH"])
    monkeypatch.delenv("MODEL_PATH")
    for key in ("HF_MODEL_REPO", "HF_MODEL_REVISION", "HF_MODEL_FILENAME", "MODEL_SHA256"):
        monkeypatch.delenv(key, raising=False)
    assets = copy.deepcopy(service.ASSETS)
    assets["model"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(service, "ASSETS", assets)
    calls = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: calls.append(kwargs) or str(path))
    service.load_model()
    service.load_model()
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "abhirath-a/digit-cnn"
    assert calls[0]["revision"] == assets["model"]["revision"]
    assets["model"]["sha256"] = "0" * 64
    service.load_model.cache_clear()
    with pytest.raises(service.AssetError, match="SHA-256"):
        service.load_model()
