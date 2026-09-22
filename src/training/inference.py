import json

import torch
from safetensors.torch import load_file

from .model import DigitCNN


def load_model(weights_path: str, config_path: str = "config.json") -> DigitCNN:
    with open(config_path) as config_file:
        config = json.load(config_file)

    input_size = config.get("input_size", 28)
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    else:
        input_size = tuple(input_size)

    model = DigitCNN(
        input_channels=config["input_channels"],
        conv_channels=tuple(config["conv_channels"]),
        num_classes=config["num_classes"],
        mlp_hidden_size=config["mlp_hidden_size"],
        input_size=input_size,
    )

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
