import torch
from safetensors.torch import load_file

from .model import DigitCNN


def load_model(weights_path: str) -> DigitCNN:
    model = DigitCNN()

    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)

    model.eval()
    return model


def predict(model: DigitCNN, image: torch.Tensor) -> tuple[int, float]:
    with torch.no_grad():
        logits = model(image.unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1)

    prediction = int(probabilities.argmax(dim=1).item())
    confidence = float(probabilities.max(dim=1).values.item())

    return prediction, confidence
