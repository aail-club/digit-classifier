# Digit Vision

An interactive computer vision demo that turns a handwritten digit into a prediction—and makes the steps behind that prediction visible.

Built with **PyTorch, OpenCV, Streamlit, and Manim**, Digit Vision connects model training, image preprocessing, inference, and visualization in one application. Users can explore built-in MNIST examples, upload an image, or take a camera snapshot.

## The experience

Each prediction becomes a short walkthrough of the classification process:

- **Start with an image:** choose an example or try your own handwriting.
- **Inspect what the model sees:** explore the grayscale image, binary crop, and final 28 × 28 input.
- **Compare predictions:** view the predicted digit alongside probabilities for all ten classes.
- **Explore the network:** watch a Manim animation illustrating representative activity for the predicted class.

The interface uses a dark theme, Inter typography, and a guided input-to-results flow to keep the technical details approachable.

## Model and image processing

The classifier is a custom convolutional neural network trained on MNIST, implemented directly in PyTorch.

Its architecture includes:

- Two convolutional blocks with 32 and 64 channels.
- Max pooling, batch normalization, and SiLU activations.
- A fully connected layer with 256 hidden units.
- Ten output classes representing digits 0–9.

An OpenCV preprocessing pipeline converts camera snapshots and uploaded images into the format expected by the network. It crops the guide region, converts the image to grayscale, applies thresholding, extracts the digit, and prepares the final model input.

Exposing these intermediate images helps explain why a prediction succeeds or fails, particularly when real photographs differ from the clean handwritten examples in MNIST.

## Network animations

The project includes ten Manim animations, individually rendered at **1080p and 60 fps** and converted into looping GIFs.

Highlighted paths are based on model activations averaged across 200 MNIST test images per digit. These animations illustrate representative class behavior rather than the exact activations of a user’s submitted image.

Pre-rendering keeps the visual experience available without requiring the deployed application to run an animation renderer.

## Engineering decisions

**Separate the interface from inference.**  
Image processing, model loading, and capture storage are independent of the Streamlit UI. This keeps the core pipeline testable and allows the interface to change without rewriting the model logic.

**Make model loading reproducible.**  
Weights are hosted on [Hugging Face](https://huggingface.co/abhirath-a/digit-cnn), downloaded from a pinned revision, and verified with a SHA-256 checksum. The application loads the checkpoint into its local architecture without executing remote Python code.

**Keep deployment lightweight.**  
The application supports CPU inference, caches the loaded model, and displays pre-rendered animations. Training and rendering dependencies remain separate from the production runtime.

**Handle Streamlit reruns deliberately.**  
Prediction results live in session state so ordinary UI interactions do not repeat inference or create duplicate capture records.

**Test the complete flow.**  
Automated checks cover preprocessing, storage, model loading, invalid inputs, and session behavior. Browser checks exercise uploads, a simulated camera, animation delivery, font loading, and mobile layout. A separate deployment check validates the hosted checkpoint against all ten built-in examples.

## Technology

| Area | Tools |
| --- | --- |
| Interface | Streamlit, Altair, CSS |
| Model training and inference | PyTorch, safetensors |
| Image processing | OpenCV, NumPy, Pillow |
| Visualization | Manim, pre-rendered GIFs |
| Model hosting | Hugging Face Hub |
| Capture storage | SQLite |
| Validation | pytest, Streamlit AppTest, Playwright |
| Deployment | Streamlit Community Cloud |
| Dependency management | uv |

## Scope and limitations

Digit Vision is an educational classification demo. Its model is trained on MNIST, so cluttered backgrounds, unusual handwriting, and poor lighting can reduce prediction quality. Confidence scores express the model’s output rather than a guarantee of correctness.

Capture records use local SQLite storage, which is not intended to provide durable persistence on an ephemeral cloud deployment.

## Run locally

With Python 3.13, uv, and Git LFS installed:

```bash
git lfs pull
uv sync --locked
uv run streamlit run app.py
