"""Pre-render digit-specific Manim GIFs. Never imported by the deployed app.

uv run --extra animation --extra training python animations/render_gifs.py
Use --digits 7 to re-render one scene. Defaults to 1920 x 1080 at 60 fps.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from manim import Text, register_font, tempconfig
from imageio_ffmpeg import get_ffmpeg_exe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "animations"))

import classification_animation as scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, nargs="+", default=list(range(10)), choices=range(10))
    args = parser.parse_args()
    output = ROOT / "assets"
    output.mkdir(exist_ok=True)
    scenes.torch.set_num_threads(min(4, os.cpu_count() or 4))
    font = ROOT / "src/digit_classifier/web/fonts/InterVariable.ttf"
    with register_font(str(font)):
        Text.set_default(font="Inter")
        for digit in args.digits:
            with tempconfig({
                "quality": "high_quality", "pixel_width": 1920, "pixel_height": 1080,
                "frame_rate": 60, "format": "mp4", "background_color": "#030712",
                "media_dir": str(ROOT / "media"), "preview": False,
                "disable_caching": True,
                "output_file": f"classification_animation_{digit}",
            }):
                scene = getattr(scenes, f"ClassificationAnimation{digit}")()
                scene.render()
                rendered = Path(scene.renderer.file_writer.movie_file_path)
                destination = output / f"classification_animation_{digit}.gif"
                temporary = output / f".classification_animation_{digit}.rendering.gif"
                palette = rendered.with_suffix(".palette.png")
                ffmpeg = get_ffmpeg_exe()
                # Two streaming passes avoid retaining an entire 1080p movie in memory.
                subprocess.run([
                    ffmpeg, "-y", "-loglevel", "error", "-i", str(rendered),
                    "-vf", "palettegen=stats_mode=diff", "-frames:v", "1", str(palette),
                ], check=True)
                subprocess.run([
                    ffmpeg, "-y", "-loglevel", "error", "-i", str(rendered), "-i", str(palette),
                    "-filter_complex", "[0:v][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                    "-loop", "0", str(temporary),
                ], check=True)
                temporary.replace(destination)
                print(f"Rendered digit {digit}: {destination} ({destination.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
