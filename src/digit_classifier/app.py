"""Gradio application with a custom HTML component to retain the original UI."""
import base64
from pathlib import Path

import gradio as gr

from . import service
from .service import classify_frame, preview_frame

WEB = Path(__file__).with_name("web")
FONT_DATA = base64.b64encode((WEB / "fonts/InterVariable.woff2").read_bytes()).decode()
PAGE_CSS = (
    "@font-face { font-family: 'Inter'; font-style: normal; font-weight: 100 900; "
    "font-display: swap; src: url(data:font/woff2;base64," + FONT_DATA + ") format('woff2'); }"
) + """
html, body, .gradio-container { background: #050914 !important; }
.gradio-container { max-width: none !important; padding: 0 !important; }
.gradio-container > .main { padding: 0 !important; }
footer { display: none !important; }
"""


# A no-argument HTML server call sends an empty list as its payload.
def app_status(_args):
    return service.app_status()


def build_app():
    gr.set_static_paths(paths=[service.ROOT / "assets"])
    with gr.Blocks(title="MNIST Vision Demo", analytics_enabled=False) as demo:
        gr.HTML(
            html_template=(WEB / "index.html").read_text(encoding="utf-8"),
            css_template=(WEB / "style.css").read_text(encoding="utf-8"),
            js_on_load=(WEB / "app.js").read_text(encoding="utf-8"),
            server_functions=[app_status, classify_frame, preview_frame],
            apply_default_css=False,
            elem_id="digit-app",
        )
    return demo


def main(demo=None):
    (demo if demo is not None else build_app()).launch(
        server_name="0.0.0.0", css=PAGE_CSS,
        # The browser only receives explicit inference results, never database files.
        show_error=False,
    )
