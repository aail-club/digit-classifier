import ssl

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# this is the only way that it works
# ty: ignore[invalid-assignment]
ssl._create_default_https_context = ssl._create_unverified_context

datasets.MNIST.mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/"]

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
