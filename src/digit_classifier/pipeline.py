import json
import os
import sys

import cv2
import numpy as np
import torch
from PySide6.QtCore import (
    QPointF,
    QPropertyAnimation,
    QRectF,
    Qt,
    QThread,
    QTimer,
    QVariantAnimation,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QImage, QMovie, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from safetensors.torch import load_file

from training import DigitCNN
from workers import CameraWorker, InferenceWorker, VisionWorker, binarize, guide_box

with open("config.json") as f:
    config = json.load(f)


DEVELOPMENT_MODE = os.environ.get("DIGIT_CLASSIFIER_ENV") == "development"
ANIMATION_PATH_TEMPLATE = os.path.join("assets", "classification_animation_{digit}.gif")

PALETTE = {
    "background": "#050914",
    "well": "#030712",
    "disabled": "#0d172a",
    "blue": "#2457e6",
    "blue_light": "#7294ff",
    "teal": "#4f9b9b",
    "yellow": "#f4c95d",
    "text": "#f5f7fc",
    "muted": "#929db2",
    "subtle": "#25314a",
}

APP_STYLES = f"""
QMainWindow, QWidget#Screen {{
    background: {PALETTE["background"]};
    color: {PALETTE["text"]};
    font-family: Inter, Avenir, Segoe UI, sans-serif;
}}
QFrame#CameraWell, QLabel#AnimationWell {{
    background: {PALETTE["well"]};
    border: 1px solid {PALETTE["subtle"]};
}}
QFrame#HorizontalRule {{
    background: {PALETTE["blue"]};
    min-height: 2px;
    max-height: 2px;
}}
QLabel#Eyebrow {{
    color: {PALETTE["yellow"]};
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
}}
QLabel#HeroTitle {{
    color: {PALETTE["text"]};
    font-size: 42px;
    font-weight: 800;
}}
QLabel#PageTitle {{
    color: {PALETTE["text"]};
    font-size: 28px;
    font-weight: 800;
}}
QLabel#SectionTitle {{
    color: {PALETTE["text"]};
    font-size: 17px;
    font-weight: 700;
}}
QLabel#BodyText {{
    color: {PALETTE["muted"]};
    font-size: 15px;
}}
QLabel#NoteText {{
    color: {PALETTE["text"]};
    border-left: 2px solid {PALETTE["yellow"]};
    padding-left: 12px;
    font-size: 12px;
}}
QLabel#StatusText {{
    color: {PALETTE["blue_light"]};
    font-size: 12px;
    font-weight: 700;
}}
QPushButton {{
    color: #090d15;
    background: {PALETTE["yellow"]};
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-size: 14px;
    font-weight: 800;
}}
QPushButton:hover {{
    background: #ffdc78;
}}
QPushButton:pressed {{
    background: #dcb044;
}}
QPushButton:disabled {{
    color: #6f7a8f;
    background: {PALETTE["disabled"]};
    border: 1px solid {PALETTE["subtle"]};
}}
"""


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


def make_curve_probabilities(
    probabilities: np.ndarray,
    neighbor_strength: float = 0.18,
    second_neighbor_strength: float = 0.035,
) -> np.ndarray:
    """
    Convert raw class probabilities into display values that produce
    nicer probability curves.

    Local maxima spread a small amount of their height into neighboring
    classes. The original probability is never decreased.

    These values are for visualization only.
    """
    values = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    display = values.copy()

    n = len(values)

    for i, value in enumerate(values):
        # Only spread actual local peaks.
        left = values[i - 1] if i > 0 else -1.0
        right = values[i + 1] if i < n - 1 else -1.0

        if value < left or value < right:
            continue

        # Immediate neighbors get a noticeable shoulder.
        if i > 0:
            display[i - 1] = max(
                display[i - 1],
                value * neighbor_strength,
            )

        if i < n - 1:
            display[i + 1] = max(
                display[i + 1],
                value * neighbor_strength,
            )

        # Second neighbors get a much smaller tail.
        if i > 1:
            display[i - 2] = max(
                display[i - 2],
                value * second_neighbor_strength,
            )

        if i < n - 2:
            display[i + 2] = max(
                display[i + 2],
                value * second_neighbor_strength,
            )

    return np.clip(display, 0.0, 1.0)


class ProbabilityCurveWidget(QWidget):
    """Draw digit probabilities as a smooth curve with visible points and labels."""

    def __init__(self):
        super().__init__()
        self.probabilities = np.zeros(10, dtype=float)
        self.setMinimumHeight(240)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

    def set_probabilities(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

        if self.probabilities.shape[0] != 10:
            self.probabilities = np.zeros(10, dtype=float)

        self.update()

    @staticmethod
    def make_curve_probabilities(
        probabilities,
        neighbor_strength=0.18,
        second_neighbor_strength=0.035,
    ):
        values = np.clip(
            np.asarray(probabilities, dtype=float),
            0.0,
            1.0,
        )

        display = values.copy()
        n = len(values)

        for i, value in enumerate(values):
            left = values[i - 1] if i > 0 else -1.0
            right = values[i + 1] if i < n - 1 else -1.0

            # Only spread local maxima.
            if value < left or value < right:
                continue

            if i > 0:
                display[i - 1] = max(
                    display[i - 1],
                    value * neighbor_strength,
                )

            if i < n - 1:
                display[i + 1] = max(
                    display[i + 1],
                    value * neighbor_strength,
                )

            if i > 1:
                display[i - 2] = max(
                    display[i - 2],
                    value * second_neighbor_strength,
                )

            if i < n - 2:
                display[i + 2] = max(
                    display[i + 2],
                    value * second_neighbor_strength,
                )

        return np.clip(display, 0.0, 1.0)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.fillRect(
            self.rect(),
            QColor(PALETTE["background"]),
        )

        rect = self.rect().adjusted(58, 24, -28, -42)

        if rect.width() <= 0 or rect.height() <= 0:
            return

        axis_pen = QPen(QColor(PALETTE["teal"]), 1)

        rule_color = QColor(PALETTE["teal"])
        rule_color.setAlpha(55)

        grid_pen = QPen(rule_color, 1)
        curve_pen = QPen(QColor(PALETTE["teal"]), 3)
        muted_text = QColor(PALETTE["muted"])

        painter.setPen(grid_pen)

        for tick in (0, 50, 100):
            y = rect.bottom() - (tick / 100) * rect.height()

            painter.drawLine(
                rect.left(),
                int(y),
                rect.right(),
                int(y),
            )

            painter.setPen(muted_text)
            painter.drawText(
                12,
                int(y) + 4,
                f"{tick}%",
            )

            painter.setPen(grid_pen)

        painter.setPen(axis_pen)

        painter.drawLine(
            rect.left(),
            rect.bottom(),
            rect.right(),
            rect.bottom(),
        )

        painter.drawLine(
            rect.left(),
            rect.top(),
            rect.left(),
            rect.bottom(),
        )

        # REAL model probabilities

        values = np.clip(
            self.probabilities,
            0.0,
            1.0,
        )

        # PRETTIFIED probabilities used ONLY for the curve
        curve_values = values

        step = rect.width() / 9

        # Actual probability locations.
        real_points = [
            QPointF(
                rect.left() + i * step,
                rect.bottom() - float(value) * rect.height(),
            )
            for i, value in enumerate(values)
        ]

        # Artificially smoothed locations for the line.
        curve_points = [
            QPointF(
                rect.left() + i * step,
                rect.bottom() - float(value) * rect.height(),
            )
            for i, value in enumerate(curve_values)
        ]

        # DRAW CURVE

        if curve_points:

            def bounded_point(point):
                return QPointF(
                    min(
                        max(point.x(), rect.left()),
                        rect.right(),
                    ),
                    min(
                        max(point.y(), rect.top()),
                        rect.bottom(),
                    ),
                )

            path = QPainterPath(curve_points[0])

            for i in range(len(curve_points) - 1):
                p0 = curve_points[max(i - 1, 0)]
                p1 = curve_points[i]
                p2 = curve_points[i + 1]
                p3 = curve_points[min(i + 2, len(curve_points) - 1)]

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

        # DRAW REAL PROBABILITY POINTS

        selected = int(np.argmax(values))

        for i, point in enumerate(real_points):
            is_selected = i == selected

            point_color = QColor(PALETTE["yellow" if is_selected else "teal"])

            painter.setBrush(point_color)
            painter.setPen(Qt.PenStyle.NoPen)

            radius = 6 if is_selected else 4

            painter.drawEllipse(
                point,
                radius,
                radius,
            )

            if is_selected:
                label_y = (
                    point.y() + 10 if point.y() < rect.top() + 24 else point.y() - 28
                )

                painter.setPen(point_color)

                painter.drawText(
                    QRectF(
                        point.x() - 28,
                        label_y,
                        56,
                        18,
                    ),
                    Qt.AlignmentFlag.AlignCenter,
                    f"{values[i] * 100:.0f}%",
                )

            painter.setPen(point_color if is_selected else muted_text)

            painter.drawText(
                QRectF(
                    point.x() - 10,
                    rect.bottom() + 8,
                    20,
                    20,
                ),
                Qt.AlignmentFlag.AlignCenter,
                str(i),
            )


def draw_digit_box(frame: np.ndarray) -> np.ndarray:
    """Draw the guide box and a best-effort digit bounding box inside it."""

    preview = frame.copy()
    x0, y0, size = guide_box(frame.shape)

    # The guide box: only pixels inside it are sent to the model.
    cv2.rectangle(
        preview,
        (x0, y0),
        (x0 + size - 1, y0 + size - 1),
        (93, 201, 244),
        3,
    )

    try:
        gray = cv2.cvtColor(
            frame[y0 : y0 + size, x0 : x0 + size],
            cv2.COLOR_BGR2GRAY,
        )
        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )
        binary = binarize(gray)
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
            (x0 + x, y0 + y),
            (x0 + x + w - 1, y0 + y + h - 1),
            (255, 180, 80),
            2,
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
        self.setMinimumSize(1024, 700)
        self.resize(1280, 800)

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

        self.setup_ui()
        self.setup_threads()

    def setup_ui(self):
        self.setStyleSheet(APP_STYLES)
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
        screen.setObjectName("Screen")
        layout = QHBoxLayout(screen)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(32)

        story = QFrame()
        story.setMinimumWidth(280)
        story.setMaximumWidth(360)
        story.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        story_layout = QVBoxLayout(story)
        story_layout.setContentsMargins(0, 12, 0, 8)
        story_layout.setSpacing(18)

        eyebrow = QLabel("DIGIT CLASSIFIER / 01")
        eyebrow.setObjectName("Eyebrow")
        story_layout.addWidget(eyebrow)

        rule = QFrame()
        rule.setObjectName("HorizontalRule")
        story_layout.addWidget(rule)

        title = QLabel("See a neural network decide.")
        title.setObjectName("HeroTitle")
        title.setWordWrap(True)
        story_layout.addWidget(title)

        copy = QLabel(
            "A single camera frame enters a compact CNN. Its convolutional layers "
            "extract visual structure; the classifier weighs all ten digits."
        )
        copy.setObjectName("BodyText")
        copy.setWordWrap(True)
        story_layout.addWidget(copy)

        invitation = QLabel(
            "You will see the prediction, confidence, and complete probability trace—not "
            "just the winning class."
        )
        invitation.setObjectName("BodyText")
        invitation.setWordWrap(True)
        story_layout.addWidget(invitation)
        story_layout.addStretch(1)

        note_text = "Capture uses the exact camera frame shown at right."
        if DEVELOPMENT_MODE:
            note_text += " The green detection box is a development-only guide."
        accent = QLabel(note_text)
        accent.setObjectName("NoteText")
        accent.setWordWrap(True)
        story_layout.addWidget(accent)

        demo = QFrame()
        demo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        demo_layout = QVBoxLayout(demo)
        demo_layout.setContentsMargins(0, 0, 0, 0)
        demo_layout.setSpacing(10)

        demo_title = QLabel("Show the camera one digit.")
        demo_title.setObjectName("PageTitle")
        demo_layout.addWidget(demo_title)

        subtitle_text = "Hold a single handwritten digit inside the yellow box."
        if DEVELOPMENT_MODE:
            subtitle_text += " Detection guide enabled."
        subtitle = QLabel(subtitle_text)
        subtitle.setObjectName("BodyText")
        subtitle.setWordWrap(True)
        demo_layout.addWidget(subtitle)

        camera_well = QFrame()
        camera_well.setObjectName("CameraWell")
        camera_well_layout = QVBoxLayout(camera_well)
        camera_well_layout.setContentsMargins(8, 8, 8, 8)
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(520, 292)
        self.camera_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet(f"color: {PALETTE['muted']};")
        camera_well_layout.addWidget(self.camera_label)
        demo_layout.addWidget(camera_well, stretch=1)

        action_row = QHBoxLayout()
        action_row.setSpacing(16)
        self.camera_status_label = QLabel("⏺ Ready")
        self.camera_status_label.setObjectName("StatusText")
        self.camera_status_label.setWordWrap(True)
        action_row.addWidget(self.camera_status_label, stretch=1)

        self.predict_button = QPushButton("Capture")
        self.predict_button.setFixedSize(190, 48)
        self.predict_button.clicked.connect(self.predict)
        action_row.addWidget(self.predict_button)
        demo_layout.addLayout(action_row)

        layout.addWidget(story, stretch=3)
        layout.addWidget(demo, stretch=7)

        return screen

    def build_results_screen(self):
        screen = QWidget()
        screen.setObjectName("Screen")
        layout = QVBoxLayout(screen)
        layout.setContentsMargins(32, 20, 32, 22)
        layout.setSpacing(10)

        header_row = QHBoxLayout()
        header_row.setSpacing(24)
        title = QLabel("Wootton Applied AI Lab (AAIL)")
        title.setObjectName("PageTitle")
        header_row.addWidget(title)
        header_row.addStretch(1)

        self.try_again_button = QPushButton("Try another digit")
        self.try_again_button.setFixedSize(180, 42)
        self.try_again_button.clicked.connect(self.return_to_camera)
        header_row.addWidget(self.try_again_button)
        layout.addLayout(header_row)

        rule = QFrame()
        rule.setObjectName("HorizontalRule")
        layout.addWidget(rule)

        result_row = QHBoxLayout()
        result_row.setSpacing(28)

        founders_panel = QFrame()
        founders_panel.setStyleSheet(
            f"background: {PALETTE['well']}; border: 1px solid {PALETTE['subtle']}; "
            "border-radius: 8px;"
        )
        founders_panel.setMinimumWidth(300)
        founders_panel.setMaximumWidth(380)
        founders_layout = QVBoxLayout(founders_panel)
        founders_layout.setContentsMargins(26, 22, 26, 22)
        founders_layout.setSpacing(10)

        founders_caption = QLabel("CO-FOUNDERS")
        founders_caption.setStyleSheet(
            f"color: {PALETTE['muted']}; font-size: 11px; font-weight: 800; "
            "letter-spacing: 2px; border: none;"
        )
        founders_layout.addWidget(founders_caption)
        founders_layout.addStretch(1)

        for founder in ("Abhi Agasanakoppa", "Arthur Khrimian"):
            founder_label = QLabel(founder)
            founder_label.setStyleSheet(
                f"color: {PALETTE['blue_light']}; font-size: 22px; "
                "font-weight: 800; border: none;"
            )
            founders_layout.addWidget(founder_label)

        founders_layout.addStretch(1)
        result_row.addWidget(founders_panel, stretch=2)

        result_panel = QFrame()
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(0, 5, 0, 2)
        result_layout.setSpacing(0)

        result_caption = QLabel("PREDICTED DIGIT")
        result_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_caption.setStyleSheet(
            f"color: {PALETTE['muted']}; font-size: 11px; font-weight: 800; "
            "letter-spacing: 2px;"
        )
        result_layout.addWidget(result_caption)

        self.prediction_label = QLabel("—")
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prediction_label.setStyleSheet(
            f"color: {PALETTE['yellow']}; font-size: 132px; font-weight: 900;"
        )
        result_layout.addWidget(self.prediction_label)

        self.confidence_label = QLabel("Waiting for prediction")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_label.setStyleSheet(
            f"color: {PALETTE['text']}; font-size: 24px; font-weight: 700;"
        )
        result_layout.addWidget(self.confidence_label)
        result_row.addWidget(result_panel, stretch=3)
        layout.addLayout(result_row)

        self.digit_color_animation = QVariantAnimation(self)
        self.digit_color_animation.setDuration(3600)
        self.digit_color_animation.setLoopCount(-1)
        for position, color in (
            (0.0, "#d7ad45"),
            (0.33, "#f4c95d"),
            (0.66, "#e5b94f"),
            (1.0, "#d7ad45"),
        ):
            self.digit_color_animation.setKeyValueAt(position, QColor(color))
        self.digit_color_animation.valueChanged.connect(self.update_digit_color)

        visual_row = QHBoxLayout()
        visual_row.setSpacing(24)

        animation_panel = QFrame()
        animation_layout = QVBoxLayout(animation_panel)
        animation_layout.setContentsMargins(0, 0, 0, 0)
        animation_layout.setSpacing(6)

        animation_caption = QLabel("How the network sees it")
        animation_caption.setObjectName("SectionTitle")
        animation_layout.addWidget(animation_caption)

        self.animation_label = QLabel()
        self.animation_label.setObjectName("AnimationWell")
        self.animation_label.setMinimumSize(430, 242)
        self.animation_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.animation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.animation_label.setWordWrap(True)
        self.animation_label.setStyleSheet(f"color: {PALETTE['muted']};")
        animation_layout.addWidget(self.animation_label, stretch=1)
        self.animation_movie = None
        self.animation_label.setText("The predicted digit animation will appear here.")

        probability_panel = QFrame()
        probability_layout = QVBoxLayout(probability_panel)
        probability_layout.setContentsMargins(0, 0, 0, 0)
        probability_layout.setSpacing(6)
        probability_title = QLabel("Probability by class")
        probability_title.setObjectName("SectionTitle")
        probability_layout.addWidget(probability_title)
        self.probability_curve = ProbabilityCurveWidget()
        self.probability_curve.setMinimumSize(300, 220)
        probability_layout.addWidget(self.probability_curve, stretch=1)

        visual_row.addWidget(animation_panel, stretch=3)
        visual_row.addWidget(probability_panel, stretch=2)
        layout.addLayout(visual_row, stretch=1)

        identity_panel = QFrame()
        identity_layout = QHBoxLayout(identity_panel)
        identity_layout.setContentsMargins(0, 2, 0, 0)
        identity_layout.setSpacing(12)

        # logo_path = os.path.join("assets", "aail-logo.svg")
        # logo = QSvgWidget(logo_path)
        # if logo.renderer().isValid():
        #     logo.setFixedSize(58, 58)
        #     identity_layout.addWidget(logo)
        # else:
        #     logo.deleteLater()

        identity_copy = QVBoxLayout()
        identity_copy.setSpacing(2)
        lab_name = QLabel("Applied AI Lab")
        lab_name.setStyleSheet(
            f"color: {PALETTE['text']}; font-size: 16px; font-weight: 800;"
        )
        identity_copy.addWidget(lab_name)

        # self.tagline_label = QLabel("From first principles to working intelligence.")
        # self.tagline_label.setStyleSheet(
        #     f"color: {PALETTE['muted']}; font-size: 13px;"
        # )
        # self.tagline_opacity = QGraphicsOpacityEffect(self.tagline_label)
        # self.tagline_label.setGraphicsEffect(self.tagline_opacity)
        # self.tagline_opacity.setOpacity(0.0)
        # self.tagline_animation = QPropertyAnimation(
        #     self.tagline_opacity,
        #     b"opacity",
        #     self,
        # )
        # self.tagline_animation.setDuration(650)
        # self.tagline_animation.setStartValue(0.0)
        # self.tagline_animation.setEndValue(1.0)
        # identity_copy.addWidget(self.tagline_label)

        identity_layout.addLayout(identity_copy)
        identity_layout.addStretch(1)
        layout.addWidget(identity_panel)

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

        display_frame = draw_digit_box(frame)
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

        # Formatting 99.97 with .1f prints "100.0%", which is never true.
        if confidence >= 99.9:
            confidence_text = ">99.9% confidence"
        else:
            confidence_text = f"{confidence:.1f}% confidence"

        # Be honest when the model is unsure (messy digit, bad lighting, no digit).
        if confidence < 60:
            confidence_text += " (not sure, try again?)"

        self.prediction_label.setText(str(prediction))
        self.confidence_label.setText(confidence_text)

        self.set_prediction_animation(prediction)
        self.probability_curve.set_probabilities(probabilities)
        # self.tagline_animation.stop()
        # self.tagline_opacity.setOpacity(0.0)

        self.processing = False
        self.camera_status_label.setText(f"Ready | Device: {self.device}")

        self.switch_screen(self.results_screen)

    @Slot(object)
    def update_digit_color(self, color):
        self.prediction_label.setStyleSheet(
            f"color: {color.name()}; font-size: 132px; font-weight: 900;"
        )

    def start_results_animations(self):
        self.digit_color_animation.stop()
        self.digit_color_animation.start()

        # self.tagline_animation.stop()
        # self.tagline_opacity.setOpacity(0.0)
        # self.tagline_animation.start()

        if self.animation_movie is not None:
            self.animation_movie.stop()
            self.animation_movie.jumpToFrame(0)
            self.scale_prediction_animation()
            self.animation_movie.start()
            QTimer.singleShot(0, self.scale_prediction_animation)

    def stop_results_animations(self):
        self.digit_color_animation.stop()
        # self.tagline_animation.stop()
        if self.animation_movie is not None:
            self.animation_movie.stop()

    def set_prediction_animation(self, prediction):
        if self.animation_movie is not None:
            self.animation_movie.stop()
            self.animation_label.clear()
            self.animation_movie.deleteLater()
            self.animation_movie = None

        animation_path = ANIMATION_PATH_TEMPLATE.format(digit=prediction)
        if os.path.exists(animation_path):
            movie = QMovie(animation_path)
            if movie.isValid():
                self.animation_movie = movie
                self.animation_label.setMovie(self.animation_movie)
                self.animation_movie.jumpToFrame(0)
                QTimer.singleShot(0, self.scale_prediction_animation)
                return

            movie.deleteLater()
            self.animation_label.setText(
                f"Animation invalid: {animation_path}\nRender and copy the digit-specific GIF assets to enable it."
            )
            return

        self.animation_label.setText(
            f"Animation missing: {animation_path}\nRender and copy the digit-specific GIF assets to enable it."
        )

    def scale_prediction_animation(self):
        if self.animation_movie is None:
            return

        frame_size = self.animation_movie.frameRect().size()
        if frame_size.isEmpty():
            return

        scaled_size = frame_size.scaled(
            self.animation_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self.animation_movie.setScaledSize(scaled_size)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "animation_movie"):
            QTimer.singleShot(0, self.scale_prediction_animation)

    @Slot(str)
    def processing_error(self, message):
        self.processing = False
        self.predict_button.setEnabled(True)
        self.camera_status_label.setText(message)

    def return_to_camera(self):
        if self.stack.currentWidget() is self.camera_screen:
            return

        self.stop_results_animations()
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
        if widget is self.results_screen:
            self.start_results_animations()

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
        self.stop_results_animations()
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
