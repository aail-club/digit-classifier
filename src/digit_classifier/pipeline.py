import json
import os
import sys

import cv2
import numpy as np
import torch
from PySide6.QtCore import QPointF, QPropertyAnimation, QRectF, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QImage, QMovie, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsOpacityEffect,
    QGroupBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from safetensors.torch import load_file

from training import DigitCNN
from workers import *

with open("config.json") as f:
    config = json.load(f)


DEVELOPMENT_MODE = os.environ.get("DIGIT_CLASSIFIER_ENV") == "development"
ANIMATION_PATH_TEMPLATE = os.path.join("assets", "classification_animation_{digit}.gif")


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


class ProbabilityCurveWidget(QWidget):
    """Draw digit probabilities as a smooth curve with visible points and labels."""

    def __init__(self):
        super().__init__()
        self.probabilities = np.zeros(10, dtype=float)
        self.setMinimumHeight(220)

    def set_probabilities(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)
        if self.probabilities.shape[0] != 10:
            self.probabilities = np.zeros(10, dtype=float)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(52, 18, -24, -42)
        if rect.width() <= 0 or rect.height() <= 0:
            return

        axis_pen = QPen(QColor("#d0d0d0"), 1)
        grid_pen = QPen(QColor("#444444"), 1)
        curve_pen = QPen(QColor("#4fc3f7"), 3)
        point_brush = QColor("#ffca28")
        text_color = QColor("#eeeeee")

        painter.fillRect(self.rect(), QColor("#151515"))
        painter.setPen(grid_pen)

        for tick in range(0, 101, 25):
            y = rect.bottom() - (tick / 100) * rect.height()
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))
            painter.setPen(text_color)
            painter.drawText(6, int(y) + 4, f"{tick}%")
            painter.setPen(grid_pen)

        painter.setPen(axis_pen)
        painter.drawLine(rect.left(), rect.bottom(), rect.right(), rect.bottom())
        painter.drawLine(rect.left(), rect.top(), rect.left(), rect.bottom())

        values = np.clip(self.probabilities, 0.0, 1.0)
        step = rect.width() / 9
        points = [
            QPointF(
                rect.left() + i * step,
                rect.bottom() - float(value) * rect.height(),
            )
            for i, value in enumerate(values)
        ]

        if points:
            def bounded_point(point):
                return QPointF(
                    min(max(point.x(), rect.left()), rect.right()),
                    min(max(point.y(), rect.top()), rect.bottom()),
                )

            path = QPainterPath(points[0])
            for i in range(len(points) - 1):
                p0 = points[max(i - 1, 0)]
                p1 = points[i]
                p2 = points[i + 1]
                p3 = points[min(i + 2, len(points) - 1)]
                c1 = bounded_point(
                    QPointF(
                        p1.x() + (p2.x() - p0.x()) / 6,
                        p1.y() + (p2.y() - p0.y()) / 6,
                    )
                )
                c2 = bounded_point(
                    QPointF(
                        p2.x() - (p3.x() - p1.x()) / 6,
                        p2.y() - (p3.y() - p1.y()) / 6,
                    )
                )
                path.cubicTo(c1, c2, p2)

            painter.setPen(curve_pen)
            painter.drawPath(path)

        painter.setPen(text_color)
        for i, point in enumerate(points):
            painter.setBrush(point_brush)
            painter.setPen(QPen(QColor("#222222"), 1))
            painter.drawEllipse(point, 5, 5)

            painter.setPen(text_color)
            painter.drawText(
                QRectF(point.x() - 22, max(rect.top(), point.y() - 26), 44, 18),
                Qt.AlignmentFlag.AlignCenter,
                f"{values[i] * 100:.0f}%",
            )
            painter.drawText(
                QRectF(point.x() - 10, rect.bottom() + 8, 20, 20),
                Qt.AlignmentFlag.AlignCenter,
                str(i),
            )


def draw_digit_box(frame: np.ndarray) -> np.ndarray:
    """Draw a best-effort digit bounding box for development previews."""

    preview = frame.copy()

    try:
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )
        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        kernel = np.ones(
            (3, 3),
            np.uint8,
        )
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
        )
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return preview

        contour = max(
            contours,
            key=cv2.contourArea,
        )

        if cv2.contourArea(contour) < 100:
            return preview

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(
            preview,
            (x, y),
            (x + w - 1, y + h - 1),
            (0, 255, 0),
            3,
        )

    except Exception:
        return preview

    return preview


class MainWindow(QMainWindow):
    process_requested = Signal(object)
    inference_requested = Signal(object)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("MNIST Vision Demo")
        self.resize(900, 700)

        self.latest_frame = None
        self.processing = False
        self.fade_animation = None

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

        self.return_timer = QTimer(self)
        self.return_timer.setSingleShot(True)
        self.return_timer.timeout.connect(self.return_to_camera)

        self.setup_ui()
        self.setup_threads()

    def setup_ui(self):
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.opacity_effect = QGraphicsOpacityEffect(self.stack)
        self.stack.setGraphicsEffect(self.opacity_effect)

        self.camera_screen = self.build_camera_screen()
        self.results_screen = self.build_results_screen()

        self.stack.addWidget(self.camera_screen)
        self.stack.addWidget(self.results_screen)

    def build_camera_screen(self):
        screen = QWidget()
        layout = QVBoxLayout(screen)

        title = QLabel("Show a handwritten digit")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 30px; font-weight: bold;")
        layout.addWidget(title)

        subtitle_text = "Camera preview"
        if DEVELOPMENT_MODE:
            subtitle_text += " (development detection overlay enabled)"

        subtitle = QLabel(subtitle_text)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 420)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("border: 1px solid #444; background: #111;")
        layout.addWidget(self.camera_label, stretch=1)

        self.camera_status_label = QLabel(f"Ready | Device: {self.device}")
        self.camera_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.camera_status_label)

        self.predict_button = QPushButton("Capture and Predict")
        self.predict_button.clicked.connect(self.predict)
        layout.addWidget(self.predict_button)

        return screen

    def build_results_screen(self):
        screen = QWidget()
        layout = QVBoxLayout(screen)

        title = QLabel("Result")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 30px; font-weight: bold;")
        layout.addWidget(title)

        self.animation_label = QLabel()
        self.animation_label.setMinimumSize(320, 180)
        self.animation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.animation_label.setStyleSheet("border: 1px solid #444; background: #111;")
        layout.addWidget(self.animation_label)

        self.animation_movie = None
        self.animation_label.setText("Digit-specific animation will appear here after prediction.")

        self.prediction_label = QLabel("—")
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prediction_label.setStyleSheet("font-size: 96px; font-weight: bold;")
        layout.addWidget(self.prediction_label)

        self.confidence_label = QLabel("Waiting for prediction")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_label.setStyleSheet("font-size: 22px;")
        layout.addWidget(self.confidence_label)

        self.probability_group = QGroupBox("Probability Curve")
        probability_layout = QVBoxLayout(self.probability_group)
        self.probability_curve = ProbabilityCurveWidget()
        probability_layout.addWidget(self.probability_curve)
        layout.addWidget(self.probability_group)

        self.try_again_button = QPushButton("Try another digit")
        self.try_again_button.clicked.connect(self.return_to_camera)
        layout.addWidget(self.try_again_button)

        return screen

    def setup_threads(self):
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker()
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.frame_ready.connect(self.update_camera)
        self.camera_worker.error.connect(self.processing_error)
        self.camera_thread.start()

        self.vision_thread = QThread()
        self.vision_worker = VisionWorker()
        self.vision_worker.moveToThread(self.vision_thread)
        self.process_requested.connect(self.vision_worker.process)
        self.vision_worker.finished.connect(self.vision_finished)
        self.vision_worker.error.connect(self.processing_error)
        self.vision_thread.start()

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

    @Slot(object)
    def update_camera(self, frame):
        self.latest_frame = frame

        if self.processing or self.stack.currentWidget() is not self.camera_screen:
            return

        display_frame = draw_digit_box(frame) if DEVELOPMENT_MODE else frame
        self.camera_label.setPixmap(
            cv_to_pixmap(
                display_frame,
                self.camera_label.size(),
            )
        )

    def predict(self):
        if self.latest_frame is None or self.processing:
            return

        captured_frame = self.latest_frame.copy()
        self.processing = True
        self.predict_button.setEnabled(False)
        self.camera_status_label.setText("Processing...")
        self.camera_label.setPixmap(
            cv_to_pixmap(
                draw_digit_box(captured_frame) if DEVELOPMENT_MODE else captured_frame,
                self.camera_label.size(),
            )
        )

        self.process_requested.emit(captured_frame)

    @Slot(object)
    def vision_finished(self, result):
        self.inference_requested.emit(result.tensor)

    @Slot(int, object)
    def inference_finished(
        self,
        prediction,
        probabilities,
    ):
        confidence = probabilities[prediction] * 100

        self.prediction_label.setText(str(prediction))
        self.confidence_label.setText(f"{confidence:.1f}% confidence")

        self.set_prediction_animation(prediction)
        self.probability_curve.set_probabilities(probabilities)

        self.processing = False
        self.camera_status_label.setText(f"Ready | Device: {self.device}")

        self.switch_screen(self.results_screen)
        self.return_timer.start(5000)

    def set_prediction_animation(self, prediction):
        if self.animation_movie is not None:
            self.animation_movie.stop()
            self.animation_label.clear()
            self.animation_movie.deleteLater()
            self.animation_movie = None

        animation_path = ANIMATION_PATH_TEMPLATE.format(digit=prediction)
        if os.path.exists(animation_path):
            self.animation_movie = QMovie(animation_path)
            self.animation_label.setMovie(self.animation_movie)
            self.animation_movie.start()
            return

        self.animation_label.setText(
            f"Animation missing: {animation_path}\nRender and copy the digit-specific GIF assets to enable it."
        )

    @Slot(str)
    def processing_error(self, message):
        self.processing = False
        self.predict_button.setEnabled(True)
        self.camera_status_label.setText(message)

    def return_to_camera(self):
        if self.stack.currentWidget() is self.camera_screen:
            return

        self.return_timer.stop()
        self.predict_button.setEnabled(True)
        self.camera_status_label.setText(f"Ready | Device: {self.device}")
        self.switch_screen(self.camera_screen)

    def switch_screen(self, widget):
        if self.stack.currentWidget() is widget:
            return

        self.fade_animation = QPropertyAnimation(
            self.opacity_effect,
            b"opacity",
            self,
        )
        self.fade_animation.setDuration(150)
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.0)
        self.fade_animation.finished.connect(lambda: self.fade_in(widget))
        self.fade_animation.start()

    def fade_in(self, widget):
        self.stack.setCurrentWidget(widget)

        self.fade_animation = QPropertyAnimation(
            self.opacity_effect,
            b"opacity",
            self,
        )
        self.fade_animation.setDuration(150)
        self.fade_animation.setStartValue(0.0)
        self.fade_animation.setEndValue(1.0)
        self.fade_animation.start()

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
