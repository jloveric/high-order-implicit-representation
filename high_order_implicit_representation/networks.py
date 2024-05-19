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
from high_order_layers_torch.sparse_optimizers import SparseLion
import torch_optimizer as alt_optim
from high_order_layers_torch.layers import high_order_fc_layers

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
            normalization=MaxAbsNormalization,  # torch.nn.LazyBatchNorm1d,
        )

        initialize_network_polynomial_layers(self.model, max_slope=1.0, max_offset=0.0)

        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def eval_step(self, batch: Tensor, name: str):
        x, y = batch
        y_hat = self(x.flatten(1))
        loss = self.loss(y_hat.flatten(), y.flatten())

        self.log(f"{name}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.eval_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, "test")

    def configure_optimizers(self):
        if self.cfg.optimizer.name == "adahessian":
            return alt_optim.Adahessian(
                self.parameters(),
                lr=self.cfg.optimizer.lr,
                betas=self.cfg.optimizer.betas,
                eps=self.cfg.optimizer.eps,
                weight_decay=self.cfg.optimizer.weight_decay,
                hessian_power=self.cfg.optimizer.hessian_power,
            )
        elif self.cfg.optimizer.name in ["adam", "sparse_lion"]:

            if self.cfg.optimizer.name == "adam":
                optimizer = optim.Adam(
                    params=self.parameters(),
                    lr=self.cfg.optimizer.lr,
                )
            elif self.cfg.optimizer.name == "sparse_lion":
                optimizer = SparseLion(
                    params=self.parameters(),
                    lr=self.cfg.optimizer.lr,
                )

            reduce_on_plateau = False
            if self.cfg.optimizer.scheduler == "plateau":
                logger.info("Reducing lr on plateau")
                lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=self.cfg.optimizer.patience,
                    factor=self.cfg.optimizer.factor,
                    verbose=True,
                )
                reduce_on_plateau = True
            elif self.cfg.optimizer.scheduler == "exponential":
                logger.info("Reducing lr exponentially")
                lr_scheduler = optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=self.cfg.optimizer.gamma
                )
            else:
                return optimizer

            scheduler = {
                "scheduler": lr_scheduler,
                "reduce_on_plateau": reduce_on_plateau,
                "monitor": "train_loss",
            }
            return [optimizer], [scheduler]
        else:
            raise ValueError(f"Optimizer {self.cfg.optimizer.name} not recognized")


class GenerativeNetwork(nn.Module):

    def __init__(
        self,
        embedding_size: int,
        input_size: int,
        output_size: int,
        mlp_width: int,
        mlp_layers: int,
        input_segments: int,
        mlp_segments: int,
        n: int,
        normalization: Any = None,
        layer_type: str = "continuous",
    ):
        super().__init__()
        self.text_layer = high_order_fc_layers(
            layer_type="polynomial",  # I want this to be polynomial or a single segment
            n=2,
            in_features=embedding_size,
            out_features=mlp_width,
        )

        self.position_layer = high_order_fc_layers(
            layer_type=layer_type,
            n=n,
            in_features=input_size,
            out_features=mlp_width,
            segments=input_segments,
        )

        self.mlp = HighOrderMLP(
            layer_type=layer_type,
            n=n,
            in_width=mlp_width,
            hidden_width=mlp_width,
            hidden_layers=mlp_layers,
            out_width=output_size,
            in_segments=mlp_segments,
            out_segments=mlp_segments,
            hidden_segments=mlp_segments,
            normalization=normalization,
        )

    def forward(self, text_embedding: Tensor, position: Tensor):
        text_out = self.text_layer(text_embedding)
        position_out = self.position_layer(position)
        x = text_out + position_out
        return self.mlp(x)


class GenNet(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.text_vector = ()
        self.position_vector = ()

        self.model = GenerativeNetwork(
            layer_type=cfg.layer_type,
            embedding_size=cfg.embedding_size,
            n=cfg.n,
            input_size=cfg.input_size,
            output_size=cfg.output_size,
            mlp_width=cfg.mlp.width,
            mlp_layers=cfg.mlp.layers,
            input_segments=cfg.input_segments,
            mlp_segments=cfg.mlp.segments,
            normalization=MaxAbsNormalization,  # torch.nn.LazyBatchNorm1d,
        )

        initialize_network_polynomial_layers(self.model, max_slope=1.0, max_offset=0.0)

        self.loss = nn.MSELoss()

    def forward(self, caption,x):
        return self.model(caption,x)

    def eval_step(self, batch: Tensor, name: str):
        #print('batch', batch)
        caption, x, color = batch
        y_hat = self(caption, x.flatten(1))
        loss = self.loss(y_hat.flatten(), color.flatten())

        self.log(f"{name}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.eval_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, "test")

    def configure_optimizers(self):
        if self.cfg.optimizer.name == "adahessian":
            return alt_optim.Adahessian(
                self.parameters(),
                lr=self.cfg.optimizer.lr,
                betas=self.cfg.optimizer.betas,
                eps=self.cfg.optimizer.eps,
                weight_decay=self.cfg.optimizer.weight_decay,
                hessian_power=self.cfg.optimizer.hessian_power,
            )
        elif self.cfg.optimizer.name in ["adam", "sparse_lion"]:

            if self.cfg.optimizer.name == "adam":
                optimizer = optim.Adam(
                    params=self.parameters(),
                    lr=self.cfg.optimizer.lr,
                )
            elif self.cfg.optimizer.name == "sparse_lion":
                optimizer = SparseLion(
                    params=self.parameters(),
                    lr=self.cfg.optimizer.lr,
                )

            reduce_on_plateau = False
            if self.cfg.optimizer.scheduler == "plateau":
                logger.info("Reducing lr on plateau")
                lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=self.cfg.optimizer.patience,
                    factor=self.cfg.optimizer.factor,
                    verbose=True,
                )
                reduce_on_plateau = True
            elif self.cfg.optimizer.scheduler == "exponential":
                logger.info("Reducing lr exponentially")
                lr_scheduler = optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=self.cfg.optimizer.gamma
                )
            else:
                return optimizer

            scheduler = {
                "scheduler": lr_scheduler,
                "reduce_on_plateau": reduce_on_plateau,
                "monitor": "train_loss",
            }
            return [optimizer], [scheduler]
        else:
            raise ValueError(f"Optimizer {self.cfg.optimizer.name} not recognized")
