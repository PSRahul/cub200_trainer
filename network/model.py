from torchvision import  models
from network.model_utils import set_parameter_requires_grad
from torchinfo import summary
#from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from torchvision import transforms
import torch


class ResNet18Model():
    def __init__(self,cfg):

        #weights = ResNet18_Weights.DEFAULT

        self.model=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=cfg["model"]["use_pretrained"])
        self.model=set_parameter_requires_grad(self.model, cfg["model"]["freeze_params"])
        self.model.fc = nn.Linear(512, cfg["model"]["num_classes"])

        
    def print_details(self):
        batch_size = 32
        summary(self.model, input_size=(batch_size, 3, 224, 224))

    def get_test_transforms(self):
        test_transforms= transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
        return test_transforms