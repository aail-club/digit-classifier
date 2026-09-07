import torch
from torch import nn


class DigitCNN(nn.Module):
    """Convolutional Neural Network (CNN) that classifies hand drawn
    digits into what numbers it is
    """

    def __init__(
        self,
        input_channels: int = 1,
        conv_channels: tuple[int, ...] = (32, 64),
        num_classes: int = 10,
        mlp_hidden_size: int = 32,
    ) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                input_channels,
                conv_channels[0],
                kernel_size=3,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(conv_channels[0]),
            nn.SiLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                conv_channels[0],
                conv_channels[1],
                kernel_size=3,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(conv_channels[1]),
            nn.SiLU(),
        )
        
        dummy = torch.zeros(1, input_channels, 28, 28)
        with torch.no_grad():
            features = self.block2(self.block1(dummy))

        flattened_size = features.flatten(1).shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(flattened_size, mlp_hidden_size),
            nn.SiLU(),
            nn.Linear(mlp_hidden_size, num_classes),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass in the neural network"""

        out = self.block1(data)
        out = self.block2(out)
        out = self.mlp(out.flatten(1))

        return out
