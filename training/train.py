import os

import torch
from dataset import loaders
from model import Digit_CNN
from torch import nn, optim


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Digit_CNN().to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    loss_fn = nn.CrossEntropyLoss()

    # 10 balances under and over fitting
    for epoch in range(10):
        running_loss = 0.0
        for idx, (features, labels) in enumerate(loaders["train"]):
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
    torch.save(model.state_dict(), os.path.join("weights/", "cnn.ptn"))


if __name__ == "__main__":
    main()
