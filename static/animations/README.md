# Pre-rendered network animations

`classification_animation_0.gif` through `classification_animation_9.gif` are rendered individually with Manim Community 0.21.0 at 1920 x 1080 and 60 fps, with an infinite playback loop. Each GIF corresponds to the predicted digit and restarts on each result.

Regenerate from the repository root:

```sh
uv sync --extra animation --extra training
uv run python animations/render_gifs.py
# Render only selected digits:
uv run python animations/render_gifs.py --digits 2 7
```

The highlights use the supplied checkpoint's activations averaged over 200 MNIST test images for each digit. They illustrate representative class behavior, rather than the activations of the current webcam frame. The renderer uses the bundled Inter font. MP4 intermediates and palette files stay in the ignored `media/` folder; only the finished GIFs are needed on Streamlit Community Cloud. Git LFS stores the GIFs.
