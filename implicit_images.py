from typing import List

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
from high_order_mlp import HighOrderMLP
from single_image_dataset import image_to_dataset
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    def __init__(self, filenames: List[str]):
        #super().__init__()
        self.output, self.input, self.image = image_to_dataset(
            filenames[0])

        """
        x = (2.0*torch.rand(1000)-1.0).view(-1, 1)
        y = (2.0*torch.rand(1000)-1.0).view(-1, 1)
        z = torch.where(x*y > 0, -0.5+0*x, 0.5+0*x)

        self.data = torch.cat([x, y], dim=1)
        self.z = z
        print(self.data.shape)

        self.transform = transform
        """

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.input[idx], self.output[idx]


class Net(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = HighOrderMLP(
            layer_type=cfg.mlp.layer_type,
            n=cfg.mlp.n,
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
        #print('full_path', full_path)    
        self.train_dataset = ImageDataset(filenames=full_path)
        self.test_dataset = ImageDataset(filenames=full_path)

    def training_step(self, batch, batch_idx):
        x, y = batch
        #print('x',x,'y',y)
        y_hat = self(x)

        loss = self.loss(y_hat, y)

        self.log(f'train_loss', loss, prog_bar=True)
        #self.log(f'train_acc', acc, prog_bar=True)

        return loss

    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=10)
        return trainloader

    def test_dataloader(self):

        testloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=10)
        return testloader

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)


@hydra.main(config_name="./config/images_config")
def run_implicit_images(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    if cfg.train is True :
        trainer = Trainer(max_epochs=cfg.max_epochs, gpus=cfg.gpus)
        model = Net(cfg)
        trainer.fit(model)
        print('testing')
        trainer.test(model)
        print('finished testing')
    else :
        # plot some data
        print('evaluating result')
        model = Net.load_from_checkpoint(cfg.checkpoint)
        model.eval()
        output, inputs, image = image_to_dataset(cfg.images[0])
        y_hat = model(inputs)
        max_x = torch.max(inputs,dim=0)
        max_y = torch.max(inputs,dim=1)
        print('x_max', max_x, 'y_max', max_y)

if __name__ == "__main__":
    run_implicit_images()
