import json
import os
from typing import TypedDict

import torch
from safetensors.torch import save_file
from torch import nn, optim

from .dataset import train_loader
from .model import DigitCNN


class ModelConfig(TypedDict):
    model_type: str
    input_channels: int
    input_size: int
    num_classes: int
    conv_channels: list[int]
    conv_kernel_size: int
    pool_kernel_size: int
    use_batch_norm: bool
    activation: str
    conv_bias: bool
    mlp_hidden_size: int

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        with open("config.json") as f:
            # config = json.load(f)
            config: ModelConfig = json.load(f) 

    except FileNotFoundError:
        config: ModelConfig = {
            "model_type": "digit_cnn",
            "input_channels": 1,
            "input_size": 28,
            "num_classes": 10,
            "conv_channels": [32, 64, 128, 256],
            "conv_kernel_size": 3,
            "pool_kernel_size": 2,
            "use_batch_norm": True,
            "activation": "SiLU",
            "conv_bias": False,
            "mlp_hidden_size": 256,
        }

    model = DigitCNN(
        input_channels=config["input_channels"],
        conv_channels=tuple(config["conv_channels"]),
        num_classes=config["num_classes"],
        mlp_hidden_size=config["mlp_hidden_size"],
    ).to(device)

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    loss_fn = nn.CrossEntropyLoss()

    # 10 balances under and over fitting
    for epoch in range(10):
        running_loss = 0.0
        for idx, (features, labels) in enumerate(train_loader):
            # move features and labels onto the device
            features = features.to(device, non_blocking=True)
            labels = labels.to(
                device, non_blocking=True
            )  # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # accumulate loss and log
            running_loss += loss.item()
            if idx % 375 == 374:
                print(
                    f"Epoch {epoch} | Steps: {idx + 1:<4} | Loss: {running_loss / 375:.3f}"
                )
                running_loss = 0.0

    if not os.path.exists("weights/"):
        os.makedirs("weights/")
    save_file(model.state_dict(), os.path.join("weights/", "model.safetensors"))
