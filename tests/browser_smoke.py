"""Run against a local app using the bundled checkpoint; camera input is synthetic.

uv run --extra browser python tests/browser_smoke.py
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--output", type=Path, default=Path("data/browser-check"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image = np.full((480, 640, 3), 255, np.uint8)
    cv2.putText(image, "7", (260, 320), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 14)
    photo = args.output / "seven.png"
    cv2.imwrite(str(photo), image)
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
        context = browser.new_context(viewport={"width": 1280, "height": 800}, permissions=["camera"])
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.goto(args.url)
        page.locator(".file-input").set_input_files(str(photo))
        expect(page.locator(".capture")).to_be_enabled()
        page.locator(".capture").click()
        expect(page.locator(".results-screen")).to_be_visible(timeout=30000)
        expect(page.locator(".prediction")).to_have_text("7")
        assert page.locator(".founders").count() == 0
        assert page.locator(".probability-table tbody tr").count() == 10
        page.screenshot(path=str(args.output / "results-desktop.png"), full_page=True)
        page.locator(".try-again").click()
        expect(page.locator(".still-frame")).to_be_visible()
        page.locator(".resume-camera").click()
        expect(page.locator(".status")).to_contain_text("Ready")
        page.wait_for_function('document.querySelector(".camera-video").readyState >= 2')
        page.screenshot(path=str(args.output / "camera-desktop.png"), full_page=True)
        page.locator(".capture").click()
        expect(page.locator(".results-screen")).to_be_visible(timeout=30000)
        expect(page.locator(".prediction")).to_have_text("7")
        assert page.locator(".camera-video").evaluate("v => v.srcObject") is None
        page.locator(".try-again").click()
        expect(page.locator(".status")).to_contain_text("Ready")
        page.wait_for_function('document.querySelector(".camera-video").readyState >= 2')
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(args.output / "camera-mobile.png"), full_page=True)
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        page.locator(".capture").click()
        expect(page.locator(".results-screen")).to_be_visible(timeout=30000)
        expect(page.locator(".prediction")).to_have_text("7")
        page.screenshot(path=str(args.output / "results-mobile.png"), full_page=True)
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        assert not errors, errors
        browser.close()
    print("Upload, camera capture, retry, camera release, probability display, and mobile checks passed.")


if __name__ == "__main__":
    main()
