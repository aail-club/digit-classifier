---
title: MNIST Vision Demo
emoji: 🔢
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
python_version: "3.13"
app_file: app.py
suggested_hardware: cpu-basic
fullWidth: true
---

# Digit classifier

A Gradio rewrite of the Applied AI Lab desktop demo. The camera and results screens retain the original colors, typography, confidence display, smooth probability trace, and digit-specific GIF playback. The founders panel has been removed and the prediction is centered. The browser captures a single unmirrored frame; only the yellow guide box is processed. OpenCV preprocessing, the CNN architecture, and the SQLite capture format are unchanged. An upload option also supports devices without a camera.

## Run locally

```sh
uv sync
uv run digit-classifier
# Or: uv run python app.py
```

Open `http://localhost:7860` and enable the camera. Remote camera access needs HTTPS; Hugging Face Spaces provides this automatically. Camera permission denial, missing cameras, invalid images, detection failures, and missing weights are shown in the interface. The camera stops on the results screen and when the page closes. “Try another digit” resumes it. Uploaded photos keep their preview when you return.

## Deploy on Hugging Face Spaces

1. Create a **Gradio Space** and upload this repository, including `src/`, `config.json`, `requirements.txt`, `app.py`, and this README. The metadata above selects the SDK, Python version, and entry point. No Qt, display server, system OpenCV packages, or GPU is required.
2. Include the supplied **`weights/model.safetensors`** when uploading. The app runs with this checkpoint by default. For alternative model storage:
   - Keep `weights/model.safetensors` with the Space. Weights and GIFs are no longer ignored by Git; `.gitattributes` uses Git LFS for them.
   - Remove the local checkpoint and set the Space variable `HF_MODEL_REPO` to your model repository (for example `your-name/digit-classifier`). The app downloads `model.safetensors` lazily on the first capture. Set `HF_MODEL_FILENAME` for a different filename, and `HF_MODEL_REVISION` to pin a commit. For a private model, set `HF_TOKEN` as a Space **secret**.
   - Set `MODEL_PATH` to an existing server-side checkpoint path.
3. Optionally upload `assets/classification_animation_0.gif` through `classification_animation_9.gif` to retain the exact original Manim animations. Without them, an animated, explicitly labeled architecture illustration is displayed. It does not represent measured activations.
4. Let the Space build, enable the camera, capture a digit, and select “Try another digit.” CPU Basic is sufficient for this small network; CUDA is selected automatically if available.

The trained checkpoint was supplied during the rewrite and is included under `weights/`; the rendered GIFs are still optional external assets. The runtime architecture is defined by **root `config.json` and `src/training/model.py`**. Any older model-repository metadata alongside the weights is ignored and is not imported by this app. This rewrite does not substitute random weights or an unrelated classifier. To train a new compatible checkpoint, run `uv sync --extra training` followed by `uv run training` (MNIST download and training happen only when explicitly invoked).

### Capture persistence

Captures are packed binary crops in `data/captures.sqlite3`, with microsecond timestamps, dimensions, prediction, and raw confidence. The database is private server-side storage and is not served as a Gradio file. Each inference uses its own SQLite connection, and concurrent inferences serialize access to the shared model. Images are not shared between browser sessions.

Space filesystem storage is ephemeral by default. If your Space has persistent storage mounted at `/data`, set `CAPTURE_DATABASE_PATH=/data/captures.sqlite3` to retain captures across restarts. Otherwise records disappear when the Space is rebuilt or its filesystem is reset. Configure retention appropriate to your deployment; the app retains the desktop demo's capture logging behavior.

## Development

```sh
uv sync --extra dev
uv run pytest
```

For browser regression checks, start the app in another terminal, then run:

```sh
uv sync --extra dev --extra browser
uv run playwright install chromium
uv run python tests/browser_smoke.py
```

This tests uploads, a synthetic webcam (no real camera access), capture, retry, camera release, probability display, and mobile layouts using the bundled checkpoint. Screenshots are written under `data/browser-check/`.

The application uses Gradio's documented custom HTML component API (`html_template`, `css_template`, `js_on_load`, and `server_functions`) rather than selectors into Gradio internals. Camera frames stay in the browser except for bounded detection previews and submitted captures. The 60% guide crop, adaptive thresholding, morphology, largest-contour selection, 20×20 resizing, 28×28 padding, and foreground centering match the desktop pipeline. The probability curve uses actual softmax values without adding probability to neighboring classes.

`src/training/` retains the training workflow; `animations/classification_animation.py` retains the optional Manim renderer (`uv sync --extra training --extra animation`). Runtime inference imports neither training datasets nor Manim. `flake.nix` provides an optional Nix development shell.
