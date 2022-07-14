from torchvision import models
from network.model_utils import set_parameter_requires_grad
from torchinfo import summary

# from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from torchvision import transforms
import torch

from torchvision.models import vit_b_16, ViT_B_16_Weights


class ResNet18Model(nn.Module):
    def __init__(self, cfg):

        # weights = ResNet18_Weights.DEFAULT
        super().__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "resnet18",
            pretrained=cfg["model"]["use_pretrained"],
        )
        self.model = set_parameter_requires_grad(
            self.model, cfg["model"]["freeze_params"]
        )
        self.model.fc = nn.Linear(512, cfg["model"]["num_classes"])

    def forward(self, x):
        return self.model.forward(x)

    def print_details(self):
        batch_size = 32
        summary(self.model, input_size=(batch_size, 3, 224, 224))

    def get_test_transforms(self):
        test_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return test_transforms

    def get_train_transforms(self):
        train_transforms = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.3, hue=0.3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return train_transforms


class ResNet50Model(nn.Module):
    def __init__(self, cfg):

        # weights = ResNet18_Weights.DEFAULT
        super().__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "resnet50",
            pretrained=cfg["model"]["use_pretrained"],
        )

        if cfg["model"]["use_pretrained"]:
            weights = ViT_B_16_Weights.DEFAULT
        else:
            weights = None

        self.model = vit_b_16(weights=weights)

        self.model = set_parameter_requires_grad(
            self.model, cfg["model"]["freeze_params"]
        )
        self.model.fc = nn.Linear(2048, cfg["model"]["num_classes"])

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.model.forward(x)
        x = self.relu(x)
        return x

    def print_details(self):
        batch_size = 32
        summary(self.model, input_size=(batch_size, 3, 224, 224))

    def get_sample_transforms(self):
        sample_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return sample_transforms

    def get_test_transforms(self):
        test_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return test_transforms

    def get_train_transforms(self):
        train_transforms = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.3, hue=0.3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return train_transforms


class ViTB16Model(nn.Module):
    def __init__(self, cfg):

        # weights = ResNet18_Weights.DEFAULT
        super().__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "resnet50",
            pretrained=cfg["model"]["use_pretrained"],
        )
        self.model = set_parameter_requires_grad(
            self.model, cfg["model"]["freeze_params"]
        )
        self.model.fc = nn.Linear(2048, cfg["model"]["num_classes"])

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.model.forward(x)
        x = self.relu(x)
        return x

    def print_details(self):
        batch_size = 32
        summary(self.model, input_size=(batch_size, 3, 224, 224))

    def get_sample_transforms(self):
        sample_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return sample_transforms

    def get_test_transforms(self):
        test_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return test_transforms

    def get_train_transforms(self):
        train_transforms = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.3, hue=0.3),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return train_transforms
