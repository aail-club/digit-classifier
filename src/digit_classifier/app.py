"""Streamlit's native widgets, with a small visual layer for the demo's identity."""
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from . import service

STYLES = Path(__file__).with_name("style.css")


def reset_capture():
    st.session_state.result = None
    st.session_state.input_epoch += 1


def apply_secrets():
    # Community Cloud secrets are optional; local bundled weights need no account.
    try:
        for key in ("MODEL_PATH", "HF_MODEL_REPO", "HF_MODEL_FILENAME", "HF_MODEL_REVISION", "HF_TOKEN", "CAPTURE_DATABASE_PATH"):
            if key in st.secrets and key not in os.environ:
                os.environ[key] = str(st.secrets[key])
    except st.errors.StreamlitSecretNotFoundError:
        pass


def steps(results=False):
    st.html(
        '<div class="flow">'
        f'<span class="{"done" if results else "active"}"><b>01</b> Capture a digit</span>'
        '<i>→</i>'
        f'<span class="{"done" if results else ""}"><b>02</b> Let the network decide</span>'
        '<i>→</i>'
        f'<span class="{"active" if results else ""}"><b>03</b> Look inside</span>'
        '</div>'
    )


def capture_screen():
    st.html('''<div class="hero"><p class="eyebrow">APPLIED AI LAB / DIGIT VISION</p>
        <h1>A little ink.<br><em>A neural network's big moment.</em></h1>
        <p>Show it a handwritten digit. Watch 784 pixels become one prediction.</p>
        <div class="hero-tags"><span>28 × 28 pixels</span><span>10 possible digits</span><span>Your image. A real CNN.</span></div></div>''')
    steps()
    left, right = st.columns([1, 1.05], gap="large")
    epoch = st.session_state.input_epoch
    payload = None
    with left, st.container(border=True, key="capture_card"):
        st.subheader("Give the network something to see")
        st.caption("Start with an example, or make it your own.")
        mode = st.radio("Image source", ["Try an example", "Upload a photo", "Use camera"],
                        horizontal=True, key=f"source_{epoch}", label_visibility="collapsed")
        if mode == "Try an example":
            digit = st.selectbox("Choose a handwritten digit", list(range(10)), index=7,
                                 format_func=lambda value: f"Digit {value}", key=f"example_{epoch}")
            payload = (service.ROOT / "static/examples" / f"digit_{digit}.png").read_bytes()
            st.caption("A real handwritten sample from the MNIST test set. Pick a digit, then run the network.")
        elif mode == "Upload a photo":
            photo = st.file_uploader("Choose a JPEG or PNG", type=["png", "jpg", "jpeg"], key=f"upload_{epoch}")
            if photo is not None:
                payload = photo.getvalue()
        else:
            st.caption("Center one digit in the middle of the frame. Use dark ink on light paper.")
            photo = st.camera_input("Take a photo", key=f"camera_{epoch}")
            if photo is not None:
                payload = photo.getvalue()
        with st.expander("A few tips for a clear prediction"):
            st.markdown("- One digit at a time, with some space around it.\n- Keep the digit inside the yellow box in the preview.\n- Use dark ink, light paper, and even lighting.")
        st.caption("Submitted captures are stored on the server for the demo. Previewing an image does not save it.")
    with right, st.container(border=True, key="preview_card"):
        st.html('<p class="eyebrow">THE INPUT</p>')
        valid = False
        if payload:
            try:
                st.image(service.preview_image(payload), width="stretch", alt="Digit preview with a yellow crop guide and blue detection box")
                st.caption("Yellow is the part the model sees. Blue marks the detected digit.")
                valid = True
            except ValueError as error:
                st.error(str(error))
        else:
            st.html('<div class="empty-preview"><span>✦</span><h3>Your digit goes here</h3><p>Take a photo or upload an image to begin.</p></div>')
        if st.button("Run the network →", type="primary", width="stretch", disabled=not valid, key="classify"):
            with st.spinner("Finding the digit, preparing 28 × 28 pixels, and running the CNN…"):
                result = service.classify_frame(payload)
            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state.result = result
                st.rerun()
    st.html('<p class="page-foot">Built to make a neural network a little less mysterious. <span>Wootton Applied AI Lab</span></p>')


def probability_chart(probabilities, digit):
    data = pd.DataFrame({
        "Digit": [str(i) for i in range(10)],
        "Probability": [float(value) * 100 for value in probabilities],
        "Winner": [i == digit for i in range(10)],
    })
    return alt.Chart(data).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("Digit:N", sort=[str(i) for i in range(10)], title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 100]), title="Probability (%)"),
        color=alt.condition(alt.datum.Winner, alt.value("#c9a54b"), alt.value("#55bbb0")),
        tooltip=[alt.Tooltip("Digit:N"), alt.Tooltip("Probability:Q", format=".2f", title="Probability (%)")],
    ).properties(height=260).configure_view(strokeWidth=0)


def results_screen(result):
    digit = result["digit"]
    probabilities = result["probabilities"]
    confidence = probabilities[digit]
    title, action = st.columns([3, 1], vertical_alignment="center")
    with title:
        st.html('<p class="eyebrow">APPLIED AI LAB / THE VERDICT</p>')
        st.title("From pixels to a prediction.")
    with action:
        st.button("← Try another digit", on_click=reset_capture, key="try_again", width="stretch")
    steps(results=True)
    summary, explanation = st.columns([1, 1.8], gap="large")
    with summary, st.container(border=True, key="prediction_card"):
        st.html(f'<div class="verdict"><p>THE NETWORK SEES A</p><div class="digit">{digit}</div><strong>{result["confidence"]}</strong></div>')
    with explanation, st.container(border=True, key="explanation_card"):
        st.subheader("Ten possibilities. One strongest signal.")
        st.write("The CNN looks for visual patterns in the image, then assigns a probability to every digit from 0 to 9.")
        if confidence < .6:
            st.warning("This one is a close call. Try a clearer digit, better lighting, or a little more space around the ink.")
        else:
            second = sorted(range(10), key=lambda index: probabilities[index], reverse=True)[1]
            st.caption(f"Next closest: {second} · {probabilities[second] * 100:.2f}% probability")
        st.progress(float(confidence), text="Model confidence in its top choice")
        st.caption("Confidence is the model's score, not a guarantee that the prediction is correct.")
    network, distribution = st.columns([1.4, 1], gap="large")
    with network:
        st.subheader("Follow the signal")
        animation = result["animation"]
        if animation:
            st.image("/" + animation, width="stretch", alt=f"Manim animation showing the neural network's representative paths for digit {digit}")
        else:
            st.info("The digit-specific animation is unavailable. The prediction above is still valid.")
        st.caption(f"Pre-rendered paths for digit {digit}, based on average activations from 200 MNIST examples. This illustrates the class, rather than this specific photo.")
    with distribution:
        st.subheader("Every digit gets a vote")
        st.altair_chart(probability_chart(probabilities, digit), width="stretch")
        st.caption("Actual softmax probabilities. Gold marks the winning class.")
    with st.expander("See how your image became 28 × 28 pixels"):
        processed = result["preprocessing"]
        cols = st.columns(3)
        for column, key, caption in zip(cols, ("grayscale", "binary", "model_input"),
                                        ("01 / Guide crop, grayscale & blur", "02 / Threshold & remove noise", "03 / Center, resize & pad")):
            with column:
                st.image(processed[key], caption=caption, clamp=True, width="stretch")
        st.caption("Only the third image is sent to the CNN. The model, checkpoint, and preprocessing are the same as the original desktop demo.")
    st.html('<p class="page-foot">A single image. A real prediction. <span>Wootton Applied AI Lab</span></p>')


def render_app():
    st.set_page_config(page_title="Digit Vision · Applied AI Lab", page_icon="✦", layout="wide", initial_sidebar_state="collapsed")
    st.html(f"<style>{STYLES.read_text(encoding='utf-8')}</style>")
    apply_secrets()
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("input_epoch", 0)
    if st.session_state.result is None:
        capture_screen()
    else:
        results_screen(st.session_state.result)
