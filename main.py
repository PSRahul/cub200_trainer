import sys
import yaml
from data.classification.data_module import ClassificationDataModule
from yaml.loader import SafeLoader
from network.model import ResNet18Model


def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=SafeLoader)

    return config


def main():
    cfg = load_config()

    model = ResNet18Model(cfg)
    data = ClassificationDataModule(
        cfg,
        train_transforms=model.get_train_transforms(),
        test_transforms=model.get_test_transforms(),
    )


if __name__ == "__main__":
    sys.exit(main())
