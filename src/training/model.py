import torch
from torch import nn


class Digit_CNN(nn.Module):
    """Convolutional Neural Network (CNN) that classifies hand drawn
    digits into what numbers it is
    """
    def __init__(self) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), bias=False),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=32),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), bias=False),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.SiLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_features=1600, out_features=32),
            nn.SiLU(),
            nn.Linear(in_features=32, out_features=10),
        )
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass in the neural network"""

        out = self.block1(data)
        out = self.block2(out)
        out = self.mlp(out.flatten(1))

        return out
