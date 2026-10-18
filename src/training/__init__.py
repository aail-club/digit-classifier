from .model import DigitCNN as DigitCNN


def main():
    # Inference must not import torchvision or download the training dataset.
    from .train import main as train

    train()

if __name__ == "__main__":
    main()
