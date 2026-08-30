import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow

from training import DigitCNN


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("hello, world!")

        label = QLabel("hello world!!!1")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.setCentralWidget(label)


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()
