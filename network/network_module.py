from sched import scheduler
from torchmetrics.functional import accuracy
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys


class ClassificationModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.train_acc = torchmetrics.Accuracy()
        # self.val_acc = torchmetrics.Accuracy()
        # self.test_acc = torchmetrics.Accuracy()
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        loss_softmax = F.softmax(y_hat, dim=1)
        self.log("train_loss", loss, prog_bar=True)
        y_acc = torch.argmax(loss_softmax, axis=1)
        # print("Class Label", y)
        # print("Model ", y_acc)
        train_acc = accuracy(y_acc, y)
        self.log("train_acc", train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        loss_softmax = F.softmax(y_hat, dim=1)
        y_acc = torch.argmax(loss_softmax, axis=1)
        val_acc = accuracy(y_acc, y)
        self.log("val_acc", val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        loss_softmax = F.softmax(y_hat, dim=0)
        y_acc = torch.argmax(loss_softmax, axis=1)
        self.train_acc(y_acc, y)
        test_acc = accuracy(y_acc, y)
        self.log("test_acc", test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        # lr=self.hparams.lr)
        return optimizer

#https://github.com/google-research/vision_transformer/issues/153