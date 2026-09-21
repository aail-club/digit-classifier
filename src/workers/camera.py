import cv2
from PySide6.QtCore import QObject, Signal, Slot


class CameraWorker(QObject):
    frame_ready = Signal(object)
    error = Signal(str)

    def __init__(self):
        super().__init__()

        self.running = True
        self.cap = cv2.VideoCapture(0)
        self.initialization_error = None

        if not self.cap.isOpened():
            self.initialization_error = "Could not open camera."

    @Slot()
    def run(self):
        if self.initialization_error:
            self.error.emit(self.initialization_error)
            return

        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                continue

            self.frame_ready.emit(frame)

    def stop(self):
        self.running = False

        if self.cap.isOpened():
            self.cap.release()
