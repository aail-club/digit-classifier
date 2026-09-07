import json
import os
import sys

import cv2
import numpy as np
import torch
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from safetensors.torch import load_file

from training import DigitCNN
from workers import *

with open("config.json") as f:
    config = json.load(f)


def cv_to_pixmap(image: np.ndarray, size=None):
    """Convert an OpenCV image to a Qt pixmap."""

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, channels = image.shape

    qimage = QImage(
        image.data,
        w,
        h,
        channels * w,
        QImage.Format.Format_RGB888,
    )

    pixmap = QPixmap.fromImage(qimage)

    if size:
        pixmap = pixmap.scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    return pixmap


# Main window
class MainWindow(QMainWindow):
    process_requested = Signal(object)
    inference_requested = Signal(object)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("MNIST Vision Demo")
        self.resize(1100, 750)

        self.latest_frame = None

        # Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DigitCNN(
            input_channels=config["input_channels"],
            conv_channels=tuple(config["conv_channels"]),
            num_classes=config["num_classes"],
            mlp_hidden_size=config["mlp_hidden_size"],
        ).to(self.device)
        tensors = load_file(os.path.join("weights", "model.safetensors"))

        self.model.load_state_dict(tensors)

        self.model.eval()

        # UI
        self.setup_ui()

        # Threads
        self.setup_threads()

    # UI
    def setup_ui(self):

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        title = QLabel("MNIST Vision Demo")
        title.setStyleSheet("font-size: 28px; font-weight: bold;")

        subtitle = QLabel("Camera → Computer Vision → Neural Network")

        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        top_layout = QGridLayout()

        camera_group = QGroupBox("Camera")

        camera_layout = QVBoxLayout(camera_group)

        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(500, 350)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        camera_layout.addWidget(self.camera_label)

        self.predict_button = QPushButton("Predict Digit")

        self.predict_button.clicked.connect(self.predict)

        camera_layout.addWidget(self.predict_button)

        top_layout.addWidget(
            camera_group,
            0,
            0,
            2,
            1,
        )

        # Pipeline
        pipeline_group = QGroupBox("Processing Pipeline")

        pipeline_layout = QVBoxLayout(pipeline_group)

        self.pipeline_labels = {}

        steps = [
            "Grayscale",
            "Binarization",
            "Noise removal",
            "Contour detection",
            "Centering",
            "Resize → 28×28",
            "CNN inference",
        ]

        for step in steps:
            label = QLabel(f"○ {step}")

            self.pipeline_labels[step] = label

            pipeline_layout.addWidget(label)

        pipeline_layout.addStretch()

        top_layout.addWidget(
            pipeline_group,
            0,
            1,
        )

        # Prediction
        prediction_group = QGroupBox("Prediction")

        prediction_layout = QVBoxLayout(prediction_group)

        self.prediction_label = QLabel("—")

        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.prediction_label.setStyleSheet("font-size: 72px; font-weight: bold;")

        prediction_layout.addWidget(self.prediction_label)

        self.confidence_label = QLabel("Waiting for prediction")

        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        prediction_layout.addWidget(self.confidence_label)

        top_layout.addWidget(
            prediction_group,
            1,
            1,
        )

        main_layout.addLayout(top_layout)

        # Image processing
        images_group = QGroupBox("Computer Vision")

        images_layout = QGridLayout(images_group)

        self.gray_label = self.make_image_label()
        self.binary_label = self.make_image_label()
        self.contour_label = self.make_image_label()
        self.resized_label = self.make_image_label()

        images_layout.addWidget(
            QLabel("Grayscale"),
            0,
            0,
        )

        images_layout.addWidget(
            QLabel("Binary"),
            0,
            1,
        )

        images_layout.addWidget(
            QLabel("Detected digit"),
            0,
            2,
        )

        images_layout.addWidget(
            QLabel("28 × 28 input"),
            0,
            3,
        )

        images_layout.addWidget(
            self.gray_label,
            1,
            0,
        )

        images_layout.addWidget(
            self.binary_label,
            1,
            1,
        )

        images_layout.addWidget(
            self.contour_label,
            1,
            2,
        )

        images_layout.addWidget(
            self.resized_label,
            1,
            3,
        )

        main_layout.addWidget(images_group)

        # Probabilities
        probability_group = QGroupBox("CNN Output")

        probability_layout = QGridLayout(probability_group)

        self.probability_bars = []

        for digit in range(10):
            label = QLabel(str(digit))

            bar = QProgressBar()

            bar.setRange(0, 100)
            bar.setValue(0)

            bar.setFormat("%v%")

            probability_layout.addWidget(
                label,
                digit,
                0,
            )

            probability_layout.addWidget(
                bar,
                digit,
                1,
            )

            self.probability_bars.append(bar)

        main_layout.addWidget(probability_group)

        # Status
        self.status_label = QLabel(f"Device: {self.device}")

        main_layout.addWidget(self.status_label)

    def make_image_label(self):

        label = QLabel()

        label.setMinimumSize(
            200,
            150,
        )

        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        label.setStyleSheet("border: 1px solid #444;")

        return label

    # Threads

    def setup_threads(self):

        # Camera
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker()

        self.camera_worker.moveToThread(self.camera_thread)

        self.camera_thread.started.connect(self.camera_worker.run)

        self.camera_worker.frame_ready.connect(self.update_camera)

        self.camera_thread.start()

        # Vision
        self.vision_thread = QThread()
        self.vision_worker = VisionWorker()

        self.vision_worker.moveToThread(self.vision_thread)

        self.process_requested.connect(self.vision_worker.process)

        self.vision_worker.finished.connect(self.vision_finished)

        self.vision_worker.error.connect(self.processing_error)

        self.vision_thread.start()

        # Inference
        self.inference_thread = QThread()

        self.inference_worker = InferenceWorker(
            self.model,
            self.device,
        )

        self.inference_worker.moveToThread(self.inference_thread)

        self.inference_requested.connect(self.inference_worker.predict)

        self.inference_worker.finished.connect(self.inference_finished)

        self.inference_worker.error.connect(self.processing_error)

        self.inference_thread.start()

    # Camera
    @Slot(object)
    def update_camera(self, frame):

        self.latest_frame = frame

        pixmap = cv_to_pixmap(
            frame,
            self.camera_label.size(),
        )

        self.camera_label.setPixmap(pixmap)

    # Prediction
    def predict(self):

        if self.latest_frame is None:
            return

        self.predict_button.setEnabled(False)

        self.prediction_label.setText("...")

        self.confidence_label.setText("Processing...")

        self.reset_pipeline()

        self.set_step(
            "Grayscale",
            "active",
        )

        self.process_requested.emit(self.latest_frame.copy())

    # Vision finished
    @Slot(object)
    def vision_finished(self, result):

        self.gray_label.setPixmap(
            cv_to_pixmap(
                result.grayscale,
                self.gray_label.size(),
            )
        )

        self.binary_label.setPixmap(
            cv_to_pixmap(
                result.binary,
                self.binary_label.size(),
            )
        )

        self.contour_label.setPixmap(
            cv_to_pixmap(
                result.cropped,
                self.contour_label.size(),
            )
        )

        # Nearest-neighbor keeps the 28x28 pixels visible.
        enlarged = cv2.resize(
            result.resized,
            (280, 280),
            interpolation=cv2.INTER_NEAREST,
        )

        self.resized_label.setPixmap(
            cv_to_pixmap(
                enlarged,
                self.resized_label.size(),
            )
        )

        # Mark CV steps complete.
        for step in [
            "Grayscale",
            "Binarization",
            "Noise removal",
            "Contour detection",
            "Centering",
            "Resize → 28×28",
        ]:
            self.set_step(
                step,
                "done",
            )

        self.set_step(
            "CNN inference",
            "active",
        )

        self.inference_requested.emit(result.tensor)

    # Inference finished
    @Slot(int, object)
    def inference_finished(
        self,
        prediction,
        probabilities,
    ):

        self.set_step(
            "CNN inference",
            "done",
        )

        self.prediction_label.setText(str(prediction))

        confidence = probabilities[prediction] * 100

        self.confidence_label.setText(f"{confidence:.1f}% confidence")

        for i, probability in enumerate(probabilities):
            value = int(probability * 100)

            self.probability_bars[i].setValue(value)

        self.status_label.setText(
            f"Prediction: {prediction} | "
            f"Confidence: {confidence:.1f}% | "
            f"Device: {self.device}"
        )

        self.predict_button.setEnabled(True)

    # Pipeline UI
    def reset_pipeline(self):

        for label in self.pipeline_labels.values():
            label.setText(label.text().replace("✓", "○").replace("→", "○"))

    def set_step(self, step, state):

        label = self.pipeline_labels[step]

        if state == "done":
            label.setText(f"✓ {step}")

        elif state == "active":
            label.setText(f"→ {step}")

        else:
            label.setText(f"○ {step}")

    # Errors
    @Slot(str)
    def processing_error(self, message):

        self.prediction_label.setText("?")

        self.confidence_label.setText(message)

        self.predict_button.setEnabled(True)

    # Cleanup
    def closeEvent(self, event):

        self.camera_worker.stop()

        self.camera_thread.quit()
        self.camera_thread.wait()

        self.vision_thread.quit()
        self.vision_thread.wait()

        self.inference_thread.quit()
        self.inference_thread.wait()

        event.accept()


def main():

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()

    sys.exit(app.exec())
