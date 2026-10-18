"""Regression coverage for the browser migration, including server transport."""
import base64
import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import gradio as gr
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from safetensors.torch import save_file

from digit_classifier import service
from digit_classifier.app import build_app
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


def test_gradio_component_transport(frame, checkpoint):
    demo = build_app()
    app = gr.mount_gradio_app(FastAPI(), demo, path="/")
    component_id = next(i for i, block in demo.blocks.items() if isinstance(block, gr.HTML))
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        for name, data in [("app_status", []), ("preview_frame", encode(frame)), ("classify_frame", encode(frame))]:
            response = client.post("/gradio_api/component_server", json={
                "component_id": component_id, "session_hash": "regression-test", "fn_name": name, "data": data,
            })
            assert response.status_code == 200, response.text
            result = response.json()
            assert "error" not in result, result
        assert client.get("/gradio_api/file=" + str(checkpoint)).status_code == 403


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
