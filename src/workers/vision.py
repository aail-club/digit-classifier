from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PySide6.QtCore import QObject, Signal, Slot


# Fraction of the shorter frame side used for the on-screen guide box.
# Only pixels inside this box are used, so shadows, table edges and hands
# outside the box can no longer be mistaken for the digit.
GUIDE_BOX_FRACTION = 0.6


def guide_box(frame_shape) -> tuple[int, int, int]:
    """Return (x, y, size) of the centered square guide box."""
    h, w = frame_shape[:2]
    size = int(min(h, w) * GUIDE_BOX_FRACTION)
    return (w - size) // 2, (h - size) // 2, size


def binarize(gray: np.ndarray) -> np.ndarray:
    """Turn a grayscale crop into white-digit-on-black, robust to uneven lighting."""
    # Adaptive thresholding picks a threshold per neighborhood instead of one
    # for the whole image, so a shadow across the paper stops mattering.
    block_size = max(3, (gray.shape[0] // 6) | 1)  # must be odd

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        10,
    )

    # Noise removal
    kernel = np.ones(
        (3, 3),
        np.uint8,
    )

    return cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )


@dataclass
class ProcessedDigit:
    grayscale: np.ndarray
    binary: np.ndarray
    cropped: np.ndarray
    centered: np.ndarray
    resized: np.ndarray
    tensor: torch.Tensor


class VisionWorker(QObject):
    finished = Signal(object)
    error = Signal(str)

    @Slot(object)
    def process(self, frame):

        try:
            # Grayscale
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            # Only keep the guide box
            x0, y0, size = guide_box(gray.shape)
            gray = gray[
                y0 : y0 + size,
                x0 : x0 + size,
            ]

            # Blur
            gray = cv2.GaussianBlur(
                gray,
                (5, 5),
                0,
            )

            # Binarization + noise removal
            binary = binarize(gray)

            # Find digit
            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if not contours:
                raise ValueError("No digit detected.")

            # Use largest contour as the digit.
            contour = max(
                contours,
                key=cv2.contourArea,
            )

            if cv2.contourArea(contour) < 100:
                raise ValueError("Digit is too small.")

            x, y, w, h = cv2.boundingRect(contour)

            # Debug image with bounding box
            contour_debug = cv2.cvtColor(
                binary,
                cv2.COLOR_GRAY2BGR,
            )

            cv2.rectangle(
                contour_debug,
                (x, y),
                (x + w - 1, y + h - 1),
                (0, 255, 0),
                3,
            )

            # Crop digit
            cropped = binary[
                y : y + h,
                x : x + w,
            ]
            
            
            # Preserve aspect ratio when putting into a square canvas
            size = max(w, h)

            centered = np.zeros(
                (size, size),
                dtype=np.uint8,
            )

            x_offset = (size - w) // 2
            y_offset = (size - h) // 2

            centered[
                y_offset : y_offset + h,
                x_offset : x_offset + w,
            ] = cropped

            # Resize digit to 20 × 20 to leave room for extra padding
            resized = cv2.resize(
                centered,
                (20, 20),
                interpolation=cv2.INTER_AREA,
            )

            # Create 28 × 28 canvas
            final = np.zeros(
                (28, 28),
                dtype=np.float32,
            )

            # Put 20×20 digit in center.
            final[
                4:24,
                4:24,
            ] = resized.astype(np.float32) / 255.0

            # Move the digit so its foreground center is centered
            ys, xs = np.where(final > 0)

            if len(xs) > 0:
                cx = xs.mean()
                cy = ys.mean()

                target_x = 13.5
                target_y = 13.5

                shift_x = round(target_x - cx)
                shift_y = round(target_y - cy)

                shifted = np.zeros_like(final)

                src_x1 = max(0, -shift_x)
                src_x2 = min(28, 28 - shift_x)

                src_y1 = max(0, -shift_y)
                src_y2 = min(28, 28 - shift_y)

                dst_x1 = max(0, shift_x)
                dst_x2 = dst_x1 + (src_x2 - src_x1)

                dst_y1 = max(0, shift_y)
                dst_y2 = dst_y1 + (src_y2 - src_y1)

                shifted[
                    dst_y1:dst_y2,
                    dst_x1:dst_x2,
                ] = final[
                    src_y1:src_y2,
                    src_x1:src_x2,
                ]

                final = shifted

            # Tensor
            tensor = torch.from_numpy(final)

            tensor = tensor.unsqueeze(0).unsqueeze(0)

            # Result
            result = ProcessedDigit(
                grayscale=gray,
                binary=binary,
                cropped=contour_debug,
                centered=centered,
                resized=resized,
                tensor=tensor,
            )

            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))
