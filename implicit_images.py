from pytorch_lightning.metrics import Metric
import os
from omegaconf import DictConfig, OmegaConf
import hydra
from pytorch_lightning.metrics.functional import accuracy
from high_order_layers_torch.layers import *
from pytorch_lightning import LightningModule, Trainer
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import torch
from .high_order_mlp import HighOrderMLP

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = datasets.ImageFolder('images', transform=transform)

class Net(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = HighOrderMLP(
            layer_type=cfg.layer_type,
            n=cfg.n,
            in_width=cfg.input.width,
            in_segments=cfg.input.segments,
            out_width=cfg.output.width,
            out_segment=cfg.output.segments,
            hidden_width=cfg.hidden.width,
            hidden_layers=cfg.hidden.layers,
            hidden_segments=cfg.hidden.segments,
        )

    def forward(self, x):
        return self.model(x)

    def setup(self, stage):
        num_train = int(self._train_fraction*40000)
        num_val = 10000
        num_extra = 40000-num_train

        train = torchvision.datasets.CIFAR100(
            root=self._data_dir, train=True, download=True, transform=transform)

        self._train_subset, self._val_subset, extra = torch.utils.data.random_split(
            train, [num_train, 10000, num_extra], generator=torch.Generator().manual_seed(1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)

        self.log(f'train_loss', loss, prog_bar=True)
        self.log(f'train_acc', acc, prog_bar=True)

        return loss

    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(
            self._train_subset, batch_size=self._batch_size, shuffle=True, num_workers=10)
        return trainloader

    def test_dataloader(self):
        testset = torchvision.datasets.CIFAR100(
            root=self._data_dir, train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=4, shuffle=False, num_workers=10)
        return testloader

    def eval_step(self, batch, batch_idx, name):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        val = self._topk_metric(logits, y)
        val = self._topk_metric.compute()

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log(f'{name}_loss', loss, prog_bar=True)
        self.log(f'{name}_acc', acc, prog_bar=True)
        self.log(f'{name}_acc5', val, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self._lr)


@hydra.main(config_name="./config/images_config")
def run_implicit_images(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    trainer = Trainer(max_epochs=cfg.max_epochs, gpus=cfg.gpus)
    model = Net(cfg)
    trainer.fit(model)
    print('testing')
    trainer.test(model)
    print('finished testing')


if __name__ == "__main__":
    run_implicit_images()
