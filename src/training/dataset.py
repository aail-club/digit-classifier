import ssl

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# this is the only way that it works
# ty: ignore[invalid-assignment]
ssl._create_default_https_context = ssl._create_unverified_context

datasets.MNIST.mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/"]

# Data augmentation: MNIST digits are perfectly centered, upright and the same
# size. Camera digits are not, so we randomly tilt, shift and resize the
# training digits to make the model robust to that.
train_transform = transforms.Compose(
    [
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
        ),
        transforms.ToTensor(),
    ]
)

# No augmentation for testing: we want to measure accuracy on real digits.
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=train_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=test_transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=512,
    shuffle=False,
)
