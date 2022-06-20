from typing import List

from pytorch_lightning.metrics import Metric
import os
from omegaconf import DictConfig, OmegaConf
import hydra
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

# from high_order_mlp import HighOrderMLP
from high_order_implicit_representation.single_image_dataset import (
    ImageNeighborhoodReader,
)
from torch.utils.data import DataLoader, Dataset
from high_order_layers_torch.networks import *


class ImageNeighborhoodDataset(Dataset):
    def __init__(self, filenames: List[str], width: int = 3, outside: int = 1):
        ind = ImageNeighborhoodReader(filenames, width=width, outside=outside)
        self.inputs = ind.features
        self.output = ind.targets
        self.image = ind.image
        self.lastx = ind.lastx
        self.lasty = ind.lasty
        self.image_neighborhood = ind

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.input[idx], self.output[idx]


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

    def setup(self, stage):

        full_path = [f"{self.root_dir}/{path}" for path in self.cfg.images]
        # print('full_path', full_path)
        self.train_dataset = ImageNeighborhoodDataset(filenames=full_path)
        self.test_dataset = ImageNeighborhoodDataset(filenames=full_path)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.log(f"train_loss", loss, prog_bar=True)
        # self.log(f'train_acc', acc, prog_bar=True)

        return loss

    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=10,
        )
        return trainloader

    def test_dataloader(self):

        testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=10,
        )
        return testloader

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)


@hydra.main(config_path="../config", config_name="neighborhood_config")
def run_implicit_neighborhood(cfg: DictConfig):
    # TODO use a space filling curve to map x,y linear coordinates
    # to space filling coordinates 1d coordinate.
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    if cfg.train is True:
        trainer = Trainer(max_epochs=cfg.max_epochs, gpus=cfg.gpus)
        model = Net(cfg)
        trainer.fit(model)
        print("testing")
        trainer.test(model)
        print("finished testing")
        print("best check_point", trainer.checkpoint_callback.best_model_path)
    else:
        # plot some data
        print("evaluating result")
        print("cfg.checkpoint", cfg.checkpoint)
        checkpoint_path = f"{hydra.utils.get_original_cwd()}/{cfg.checkpoint}"
        print("checkpoint_path", checkpoint_path)
        model = Net.load_from_checkpoint(checkpoint_path)
        model.eval()
        image_dir = f"{hydra.utils.get_original_cwd()}/{cfg.images[1]}"
        # output, inputs, image = image_to_dataset(
        #    image_dir, rotations=cfg.rotations)
        print("image_dir", image_dir)
        ind = ImageNeighborhoodDataset(image_dir, width=3, outside=1)
        inputs = ind.inputs
        image = ind.image
        lastx = ind.lastx
        lasty = ind.lasty

        num_batches = 1000
        length = len(inputs)
        batch_size = length // num_batches
        accum = []
        for j in range(0, num_batches):
            batch = inputs[j * length : (j + 1) * length]
            y_hat = model(batch)
            accum += y_hat

        y_hat = torch.stack(accum)

        max_x = torch.max(inputs, dim=0)
        max_y = torch.max(inputs, dim=1)
        print("x_max", max_x, "y_max", max_y)
        print("y_hat.shape", y_hat.shape)
        print("image.shape", image.shape)
        ans = y_hat.reshape(lastx, lasty, image.shape[2], -1)
        ans = (ans + 1.0) / 2.0
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(ans.detach().numpy()[:, :, :, 0])
        axarr[0].set_title("fit")
        axarr[1].imshow(image)
        axarr[1].set_title("original")
        plt.show()


if __name__ == "__main__":
    run_implicit_neighborhood()
