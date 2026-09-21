import torch
from torch import nn


class DigitCNN(nn.Module):
    """Convolutional neural network for image classification."""

    def __init__(
        self,
        input_channels: int = 1,
        conv_channels: tuple[int, ...] = (32, 64),
        num_classes: int = 10,
        mlp_hidden_size: int = 32,
        input_size: tuple[int, int] = (28, 28),
    ) -> None:
        super().__init__()

        if not conv_channels:
            raise ValueError("conv_channels must contain at least one channel size")

        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.num_classes = num_classes
        self.mlp_hidden_size = mlp_hidden_size
        self.input_size = input_size

        blocks = []
        in_channels = input_channels

        for out_channels in conv_channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        bias=False,
                    ),
                    nn.MaxPool2d(kernel_size=2),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(),
                )
            )

            in_channels = out_channels

        self.features = nn.Sequential(*blocks)

        dummy = torch.zeros(
            1,
            input_channels,
            *input_size,
        )

        with torch.no_grad():
            features = self.features(dummy)

        flattened_size = features.flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, mlp_hidden_size),
            nn.SiLU(),
            nn.Linear(mlp_hidden_size, num_classes),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the network."""
        out = self.features(data)
        out = out.flatten(1)
        return self.classifier(out)
