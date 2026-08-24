import os

import torch
from safetensors.torch import save_file
from torch import nn, optim

from .dataset import train_loader
from .model import Digit_CNN


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Digit_CNN().to(device)
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


