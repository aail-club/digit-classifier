"""Streamlit browser regression test using synthetic camera input.

Start `streamlit run app.py`, then run this script with the browser extra.
"""
import argparse
import re
from pathlib import Path

import cv2
from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--output", type=Path, default=Path("data/browser-check"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    photo = Path(__file__).resolve().parents[1] / "static/examples/digit_7.png"
    image = cv2.imread(str(photo))
    video = args.output / "seven.y4m"
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).tobytes()
    with video.open("wb") as output:
        output.write(b"YUV4MPEG2 W640 H480 F30:1 Ip A1:1 C420jpeg\n")
        for _ in range(60):
            output.write(b"FRAME\n" + yuv)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={video.resolve()}",
        ])
        context = browser.new_context(viewport={"width": 1440, "height": 1050}, permissions=["camera"])
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        expect(page.get_by_role("button", name="Run the network →")).to_be_enabled(timeout=30000)
        page.wait_for_function("[...document.fonts].some(f => f.family === 'Inter' && f.status === 'loaded')")
        page.screenshot(path=str(args.output / "capture-desktop.png"), full_page=True)
        page.get_by_role("button", name="Run the network →").click()
        expect(page.locator(".verdict .digit")).to_have_text("7", timeout=30000)
        gif = page.locator('img[alt*="Manim animation"]')
        expect(gif).to_be_visible()
        expect(gif).to_have_attribute("src", re.compile(r"/app/static/animations/classification_animation_7\.gif$"))
        expect(gif).to_have_js_property("naturalWidth", 1920)
        page.locator('[data-testid="stVegaLiteChart"] svg.marks').wait_for(state="visible")
        page.screenshot(path=str(args.output / "results-desktop.png"), full_page=True)
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        page.get_by_role("button", name="← Try another digit").click()
        page.get_by_text("Upload a photo", exact=True).click()
        page.locator('input[type="file"]').set_input_files(str(photo))
        page.get_by_role("button", name="Run the network →").click()
        expect(page.locator(".verdict .digit")).to_have_text("7", timeout=30000)
        page.get_by_role("button", name="← Try another digit").click()
        page.get_by_text("Use camera", exact=True).click()
        enable = page.get_by_role("button", name="Enable camera", exact=True)
        if enable.is_visible():
            enable.click()
        page.get_by_role("button", name=re.compile("^Take photo$", re.IGNORECASE)).click(timeout=15000)
        page.get_by_role("button", name="Run the network →").click()
        expect(page.locator(".verdict .digit")).to_have_text("7", timeout=30000)
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(args.output / "results-mobile.png"), full_page=True)
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        page.get_by_text("Every digit gets a vote", exact=True).scroll_into_view_if_needed()
        expect(page.locator('[data-testid="stVegaLiteChart"] svg.marks')).to_be_visible()
        page.screenshot(path=str(args.output / "chart-mobile.png"), full_page=True)
        page.get_by_role("button", name="← Try another digit").click()
        expect(page.get_by_role("button", name="Run the network →")).to_be_enabled()
        page.screenshot(path=str(args.output / "capture-mobile.png"), full_page=True)
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        assert not errors, errors
        browser.close()
    print("Example, upload, synthetic camera, reset, GIF, Inter, and mobile checks passed.")


if __name__ == "__main__":
    main()
