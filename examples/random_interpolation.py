from typing import List

import os
from omegaconf import DictConfig, OmegaConf
import hydra
from torchmetrics.functional import accuracy
from high_order_layers_torch.layers import *
from high_order_layers_torch.networks import *
from pytorch_lightning import LightningModule, Trainer
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from high_order_implicit_representation.random_sample_dataset import (
    RandomImageSampleDataModule,
)

# from high_order_mlp import HighOrderMLP
from high_order_implicit_representation.single_image_dataset import image_to_dataset
from torch.utils.data import DataLoader, Dataset
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)


class Net(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model = HighOrderMLP(
            layer_type=cfg.mlp.layer_type,
            n=cfg.mlp.n,
            n_in=cfg.mlp.n_in,
            n_hidden=cfg.mlp.n_in,
            n_out=cfg.mlp.n_out,
            in_width=cfg.mlp.input.width,
            in_segments=cfg.mlp.input.segments,
            out_width=cfg.mlp.output.width,
            out_segments=cfg.mlp.output.segments,
            hidden_width=cfg.mlp.hidden.width,
            hidden_layers=cfg.mlp.hidden.layers,
            hidden_segments=cfg.mlp.hidden.segments,
        )
        self.root_dir = f"{hydra.utils.get_original_cwd()}"
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat, y)

        self.log(f"train_loss", loss, prog_bar=True)
        # self.log(f'train_acc', acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)


@hydra.main(config_path="../config", config_name="random_interpolation")
def run_implicit_images(cfg: DictConfig):

    logger.info(OmegaConf.to_yaml(cfg))
    logger.info(f"Working directory : {os.getcwd()}")
    logger.info(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    if cfg.train is True:

        datamodule = RandomImageSampleDataModule(
            image_size=cfg.image_size,
            folder=cfg.folder,
            num_target_pixels=cfg.num_target_pixels,
            num_feature_pixels=cfg.num_feature_pixels,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            split_frac=cfg.split_frac,
        )

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        early_stopping = EarlyStopping(monitor="train_loss", patience=10)
        trainer = Trainer(
            callbacks=[early_stopping, lr_monitor],
            max_epochs=cfg.max_epochs,
            gpus=cfg.gpus,
            gradient_clip_val=cfg.gradient_clip,
        )

        trainer = Trainer(max_epochs=cfg.max_epochs, gpus=cfg.gpus)
        model = Net(cfg)
        trainer.fit(model, datamodule=datamodule)

        logger.info("testing")
        trainer.test(model, datamodule=datamodule)
        logger.info("finished testing")
        logger.info(f"best check_point {trainer.checkpoint_callback.best_model_path}")
    else:
        # plot some data
        logger.info("evaluating result")
        logger.info(f"cfg.checkpoint {cfg.checkpoint}")
        checkpoint_path = f"{hydra.utils.get_original_cwd()}/{cfg.checkpoint}"
        logger.info("checkpoint_path {checkpoint_path}")
        model = Net.load_from_checkpoint(checkpoint_path)
        model.eval()
        image_dir = f"{hydra.utils.get_original_cwd()}/{cfg.images[0]}"
        output, inputs, image = image_to_dataset(image_dir, rotations=cfg.rotations)
        y_hat = model(inputs)
        max_x = torch.max(inputs, dim=0)
        max_y = torch.max(inputs, dim=1)
        print("x_max", max_x, "y_max", max_y)
        print("y_hat.shape", y_hat.shape)
        print("image.shape", image.shape)
        ans = y_hat.reshape(image.shape[0], image.shape[1], image.shape[2])
        ans = (ans + 1.0) / 2.0
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(ans.detach().numpy())
        axarr[0].set_title("fit")
        axarr[1].imshow(image)
        axarr[1].set_title("original")
        plt.show()


if __name__ == "__main__":
    run_implicit_images()
