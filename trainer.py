from asyncio.log import logger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import sys
import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datetime import datetime
from pytorch_lightning.callbacks import LearningRateMonitor
import optuna
import joblib
import pathlib

now = datetime.now()


class LightningTrainer:
    def __init__(self, cfg):
        date_save_string = now.strftime("%d%m%Y_%H%M")

        self.checkpoint_dir = os.path.join(
            cfg["trainer"]["checkpoint_dir"], date_save_string
        )
        pathlib.Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        logger = TensorBoardLogger(self.checkpoint_dir, name=cfg["trainer"]["exp_name"])

        lr_monitor = LearningRateMonitor(logging_interval="step")

        self.trainer = pl.Trainer(
            enable_checkpointing=True,
            logger=logger,
            accelerator="gpu",
            devices=1,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=5)
            ],
            # ],  # lr_monitor],
            max_epochs=cfg["trainer"]["max_epochs"],
            default_root_dir=self.checkpoint_dir,
            log_every_n_steps=1,
        )

    def tune_learning_rate(self, model, data):

        lr_finder = self.trainer.tuner.lr_find(
            model=model,
            train_dataloaders=data.train_dataloader(),
            val_dataloaders=data.val_dataloader(),
        )

        model.hparams.lr = lr_finder.suggestion()
        print("Tuning Learning Rate", model.hparams.lr)

    def tune(self, model):
        self.trainer.tune(model)

    def train(self, model, data):
        self.trainer.fit(
            model=model,
            train_dataloaders=data.train_dataloader(),
            val_dataloaders=data.val_dataloader(),
        )

    def optuna_tune(self, model, data, num_trials):
        lr_study = optuna.create_study(direction="minimize")
        lr_study.optimize(
            lambda trial: self.optuna_objective(trial, model=model, data=data),
            n_trials=num_trials,
        )
        joblib.dump(lr_study, os.path.join(self.checkpoint_dir, "study.pkl"))
        print("Model Tuned : Best Learning Rate", lr_study.best_params["learning_rate"])
        return lr_study.best_params["learning_rate"]

    def optuna_objective(self, trial, model, data):
        lr = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
        model.hparams.lr = lr
        trainer = pl.Trainer(
            enable_checkpointing=False,
            logger=False,
            accelerator="gpu",
            devices=1,
            max_epochs=10,
        )
        trainer.fit(
            model=model,
            train_dataloaders=data.train_dataloader(),
        )

        return trainer.callback_metrics["train_loss"]
