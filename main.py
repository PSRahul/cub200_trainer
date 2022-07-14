import sys
import tracemalloc
import yaml
from data.classification.data_module import ClassificationDataModule
from yaml.loader import SafeLoader
from network.model import ResNet18Model, ResNet50Model, ViTB16Model
from trainer import LightningTrainer
from network.network_module import ClassificationModel
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, default="config.yaml")
    args = parser.parse_args()
    return args


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=SafeLoader)

    return config


def main():

    args = get_args()
    cfg = load_config(args.c)
    pytorch_model = ResNet50Model(cfg)

    if cfg["debug"]:
        print("Debug Mode Enabled")
        train_transforms = pytorch_model.get_sample_transforms()
        test_transforms = pytorch_model.get_sample_transforms()
    else:
        train_transforms = pytorch_model.get_train_transforms()
        test_transforms = pytorch_model.get_test_transforms()

    data = ClassificationDataModule(
        config=cfg, train_transforms=train_transforms, test_transforms=test_transforms
    )
    data.setup()
    model = ClassificationModel(pytorch_model)
    trainer = LightningTrainer(cfg=cfg)

    if cfg["tune"]["enable"]:
        model.hparams.lr = trainer.optuna_tune(model, data, cfg["tune"]["num_trials"])
    trainer.train(model, data)


if __name__ == "__main__":
    sys.exit(main())
