import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

import os

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        
        train_dataset=ImageFolder(root=os.path.join(self.cfg["data"]["root_folder"],self.cfg["data"]["train_folder"]))
        val_dataset=ImageFolder(root=os.path.join(self.cfg["data"]["root_folder"],self.cfg["data"]["val_folder"]))
        test_dataset=ImageFolder(root=os.path.join(self.cfg["data"]["root_folder"],self.cfg["data"]["test_folder"]))

        print("Datasets are Accessible")            
    
    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset=ImageFolder(root=os.path.join(self.cfg["data"]["root_folder"],self.cfg["data"]["train_folder"]))
            self.val_dataset=ImageFolder(root=os.path.join(self.cfg["data"]["root_folder"],self.cfg["data"]["val_folder"]))

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = ImageFolder(root=os.path.join(self.cfg["data"]["root_folder"],self.cfg["data"]["test_folder"]))

        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32)

    