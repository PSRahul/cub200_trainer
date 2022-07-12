import sys
import yaml
from data.classification.data_module import ClassificationDataModule
from yaml.loader import SafeLoader
from network.model import ResNet18Model
from trainer import LightningTrainer
from network.network_module import ClassificationModel


def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=SafeLoader)

    return config


def main():
    cfg = load_config()

    pytorch_model = ResNet18Model(cfg)
    data = ClassificationDataModule(
        config=cfg,
        train_transforms=pytorch_model.get_train_transforms(),
        test_transforms=pytorch_model.get_test_transforms(),
    )
    data.setup()
    model = ClassificationModel(pytorch_model)
    trainer = LightningTrainer(cfg=cfg)
    trainer.tune_learning_rate(model, data)


if __name__ == "__main__":
    sys.exit(main())
