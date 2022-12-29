
# ⚡Lightning Fast Trainer⚡

This repository provides recipes in **PyTorch Lightning** for task of image classification.

## Features

1. Fully functional network and data modules configurable by the custom configuration files
2. Tensorboard Logging
3. Automatic Learning Rate Optimisation with Optuna

## Instructions

### Setting up an Conda Environment

After installing Ananconda run `conda create -n trainer python=3.8` and install the dependencies using `pip install -r requirements.txt`.

### Model Training

To start a classification training run `python main.py -c config.yaml`

### Image Classification
1. ResNet18
2. ResNet50
3. ViT B16
