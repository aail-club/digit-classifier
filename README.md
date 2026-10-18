# Digit Vision · Applied AI Lab

A Streamlit demo that turns a handwritten digit into a prediction, an animated tour of the network, and an explanation of the model input. Built-in MNIST examples make the first run immediate; photo uploads and the browser camera let visitors try their own handwriting.

The original CNN, trained checkpoint, 60% guide crop, image preprocessing, confidence wording, and SQLite capture format are retained. The model weights download from [abhirath-a/digit-cnn](https://huggingface.co/abhirath-a/digit-cnn) and are not stored in this GitHub checkout. The existing Manim GIF and Inter setup is unchanged. The founders panel remains removed.

## Run locally

```sh
uv sync
uv run streamlit run app.py
# Or: uv run digit-classifier
```

Run from the repository root so Streamlit reads `.streamlit/config.toml`. Open http://localhost:8501, choose an example or a photo, and select **Run the network**. Results show the predicted digit, confidence, ten class probabilities, the matching Manim GIF, and expandable preprocessing images. **Try another digit** clears the result and resets input widgets. Reruns do not repeat inference or save duplicate captures.

The camera uses Streamlit's native snapshot widget. Center a digit in the frame, take a photo, then check the yellow guide in the preview before running the network. This version does not draw overlays on the live camera feed. Remote camera access requires HTTPS; Community Cloud provides it. Upload and examples work when camera access is unavailable.

## Streamlit Community Cloud

1. Push the **streamlit-rewrite** branch to GitHub, including the existing GIF assets. Do not add a safetensors file: the app fetches it from Hugging Face. Streamlit Community Cloud supports Git LFS for the existing GIFs. After a local clone, use `git lfs pull` if necessary.
2. At [share.streamlit.io](https://share.streamlit.io), create an app and select your GitHub repository, branch **streamlit-rewrite**, and main file **app.py**.
3. In Advanced settings, select **Python 3.13**, matching local development, and deploy. No secrets, system packages, GPU, training-data downloads, or animation renderer are needed. The first prediction downloads the small checkpoint; later predictions reuse the cached file and loaded model.

Community Cloud uses the root **uv.lock** and **pyproject.toml** to install the runtime. Linux is explicitly locked to the CPU-only PyTorch wheel through the official PyTorch CPU index, avoiding CUDA downloads. `requirements.txt` has been removed to keep one deployment dependency definition. Training, rendering, and test dependencies are optional extras and are not installed in production.

`.streamlit/config.toml` enables static serving, an 8 MB upload limit, the dark theme, and the bundled Inter font. Only `static/` is publicly served; the SQLite database and model weights are outside it. The animations are served as GIF files without being base64-encoded into each rerun. Camera images and results stay in each visitor's session; the shared model is loaded once and inference is serialized.

Deployment references: [Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [dependency precedence](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies), [Git LFS support](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).

### Model hosting and validation

The model is configured in **deployment-assets.json**:

- Repository: `abhirath-a/digit-cnn`
- File: `model.safetensors`
- Revision: `c945f6dfe72db3c8a2219fbfd9e8161c07350361`
- SHA-256: `c01144b88c53199ea58702c00511fd2e8339c22f6c2cb30719943b176b4412b1`

`hf_hub_download` downloads to the Hugging Face cache, and the loader verifies SHA-256 before loading the model. Only the weights are downloaded; the app never executes `model.py` or `inference.py` from the Hub. A changed remote `main` branch does not change the pinned deployment. Cold starts or Cloud rebuilds may need to download the file again because local storage is ephemeral.

Before deployment, run:

```sh
uv run python scripts/check_deployment.py
```

This checks the remote repository, downloads the production weights, verifies the checksum and architecture, and classifies all ten built-in examples. It ignores local `MODEL_PATH` during validation and uses a temporary database, so it cannot pass just because a local checkpoint exists or add records to the demo's database. The GitHub Actions workflow runs tests and this remote-model check on pushes and pull requests.

For optional overrides, use `HF_MODEL_REPO`, `HF_MODEL_FILENAME`, `HF_MODEL_REVISION`, and `MODEL_SHA256` as environment variables or Community Cloud secrets; see `.streamlit/secrets.toml.example`. The public default model needs **no token**. A private model needs a read-only `HF_TOKEN` in Streamlit secrets. `MODEL_PATH` explicitly selects a local file for development; an invalid path produces an error, so remove this setting to restore remote loading. Do not commit tokens or `.streamlit/secrets.toml`.

The runtime architecture is defined by root `config.json` and `src/training/model.py`. If you deploy new weights, update the pinned revision/checksum and rerun the deployment check. `weights/` is ignored by Git. Existing local copies are preserved; removing a file from the current branch does not purge historical Git LFS objects from GitHub. No Git history is rewritten by this change.

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
