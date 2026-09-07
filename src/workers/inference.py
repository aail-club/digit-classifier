import torch
from PySide6.QtCore import QObject, Signal, Slot


class InferenceWorker(QObject):
    finished = Signal(int, object)
    error = Signal(str)

    def __init__(self, model, device):
        super().__init__()

        self.model = model
        self.device = device

        self.model.eval()

    @Slot(object)
    def predict(self, tensor):

        try:
            tensor = tensor.to(self.device)

            with torch.inference_mode():
                logits = self.model(tensor)

                probabilities = torch.softmax(
                    logits,
                    dim=1,
                )

            prediction = probabilities.argmax(dim=1).item()

            probabilities = probabilities[0].cpu().numpy()

            self.finished.emit(
                prediction,
                probabilities,
            )

        except Exception as e:
            self.error.emit(str(e))
