from syslog import LOG_SYSLOG
from torchmetrics.functional import accuracy
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class ClassificationModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.train_acc = torchmetrics.Accuracy()
        # self.val_acc = torchmetrics.Accuracy()
        # self.test_acc = torchmetrics.Accuracy()
        self.save_hyperparameters()

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        loss_softmax = F.softmax(y_hat)
        y_acc = torch.argmax(loss_softmax, axis=1)
        train_acc = accuracy(y_acc, y)
        self.log("train_acc", train_acc)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        loss_softmax = F.softmax(y_hat)
        y_acc = torch.argmax(loss_softmax, axis=1)
        val_acc = accuracy(y_acc, y)
        self.log("val_acc", val_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        loss_softmax = F.softmax(y_hat)
        y_acc = torch.argmax(loss_softmax, axis=1)
        self.train_acc(y_acc, y)
        test_acc = accuracy(y_acc, y)
        self.log("test_acc", test_acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())
