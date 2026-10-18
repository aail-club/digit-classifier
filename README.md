# Digit Vision · Applied AI Lab

A Streamlit demo that turns a handwritten digit into a prediction, an animated tour of the network, and an explanation of the model input. Built-in MNIST examples make the first run immediate; photo uploads and the browser camera let visitors try their own handwriting.

The original CNN, trained checkpoint, 60% guide crop, image preprocessing, confidence wording, and SQLite capture format are retained. All ten high-quality Manim GIFs and the Inter variable font are bundled. The founders panel remains removed.

## Run locally

```sh
uv sync
uv run streamlit run app.py
# Or: uv run digit-classifier
```

Run from the repository root so Streamlit reads `.streamlit/config.toml`. Open http://localhost:8501, choose an example or a photo, and select **Run the network**. Results show the predicted digit, confidence, ten class probabilities, the matching Manim GIF, and expandable preprocessing images. **Try another digit** clears the result and resets input widgets. Reruns do not repeat inference or save duplicate captures.

The camera uses Streamlit's native snapshot widget. Center a digit in the frame, take a photo, then check the yellow guide in the preview before running the network. This version does not draw overlays on the live camera feed. Remote camera access requires HTTPS; Community Cloud provides it. Upload and examples work when camera access is unavailable.

## Streamlit Community Cloud

1. Push the **streamlit-rewrite** branch to GitHub, including the Git LFS objects for `weights/model.safetensors` and `static/animations/*.gif`. Streamlit Community Cloud supports Git LFS and downloads these files automatically. After a local clone, use `git lfs pull` if necessary.
2. At [share.streamlit.io](https://share.streamlit.io), create an app and select your GitHub repository, branch **streamlit-rewrite**, and main file **app.py**.
3. In Advanced settings, select **Python 3.13**, matching local development, and deploy. No secrets, system packages, GPU, training-data downloads, or animation renderer are needed for the bundled demo.

Community Cloud uses the root **uv.lock** and **pyproject.toml** to install the runtime. Linux is explicitly locked to the CPU-only PyTorch wheel through the official PyTorch CPU index, avoiding CUDA downloads. `requirements.txt` has been removed to keep one deployment dependency definition. Training, rendering, and test dependencies are optional extras and are not installed in production.

`.streamlit/config.toml` enables static serving, an 8 MB upload limit, the dark theme, and the bundled Inter font. Only `static/` is publicly served; the SQLite database and model weights are outside it. The animations are served as GIF files without being base64-encoded into each rerun. Camera images and results stay in each visitor's session; the shared model is loaded once and inference is serialized.

Deployment references: [Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [dependency precedence](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies), [Git LFS support](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).

### Optional model settings

The supplied checkpoint works by default. To use a different compatible checkpoint, set `MODEL_PATH`, or use `HF_MODEL_REPO` with optional `HF_MODEL_FILENAME`, `HF_MODEL_REVISION`, and `HF_TOKEN`. These can be environment variables or Community Cloud secrets; see `.streamlit/secrets.toml.example`. For Hub loading, remove the bundled local checkpoint or point `MODEL_PATH` to a nonexistent path so the Hub fallback is used. Do not commit tokens or `.streamlit/secrets.toml`.

The runtime architecture is defined by root `config.json` and `src/training/model.py`. Older metadata beside the supplied checkpoint is ignored.

### Captures and persistence

Submitted captures are stored as packed binary crops in `data/captures.sqlite3`, with microsecond timestamps, dimensions, prediction, and raw confidence. Each inference owns its SQLite connection. Previewing a photo or changing an example does not save it. Community Cloud local storage is ephemeral, so this database is for demonstration rather than durable recordkeeping. `CAPTURE_DATABASE_PATH` changes its location; changing the path does not make Cloud storage persistent.

## Tests

```sh
uv sync --extra dev --extra browser
uv run pytest
# With the app running in another terminal:
uv run playwright install chromium
uv run python tests/browser_smoke.py
```

Tests cover preprocessing, storage, concurrent inference, checkpoint predictions, invalid inputs, Streamlit session reruns and reset, and all ten complete 1080p GIFs. Browser checks cover the example flow, uploads, a synthetic webcam, GIF delivery, Inter loading, and mobile layout. They do not access a real camera. Screenshots go under ignored `data/browser-check/`.

## Regenerate animations or train

```sh
uv sync --extra animation --extra training
uv run python animations/render_gifs.py
# Or a single digit:
uv run python animations/render_gifs.py --digits 7
# Train a new compatible checkpoint:
uv run training
```

Each animation is individually rendered at 1920 × 1080 and 60 fps, then converted into a looping GIF. Highlighted paths are based on model activations averaged over 200 MNIST test images per digit; they describe representative class behavior, not the current camera capture. The renderer and its dataset downloads are never imported by the app. Intermediate movies remain in ignored `media/`.

Inter is served locally from `static/fonts/`; its SIL Open Font License is included there. Built-in examples are derived from MNIST test images; source indices are listed in `static/examples/README.md`.
