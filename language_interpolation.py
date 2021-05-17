from typing import List

import os
from omegaconf import DictConfig, OmegaConf
import hydra
from pytorch_lightning.metrics.functional import accuracy
from high_order_layers_torch.layers import *
from pytorch_lightning import LightningModule, Trainer
import torch.optim as optim
import torch
from high_order_layers_torch.networks import *
from single_text_dataset import SingleTextDataset
from torchsummary import summary
from single_text_dataset import dataset_from_file, encode_input_from_text, decode_output_to_text, ascii_to_float
import random

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
            out_width=128,  # ascii has 128 characters
            out_segments=cfg.mlp.output.segments,
            hidden_width=cfg.mlp.hidden.width,
            hidden_layers=cfg.mlp.hidden.layers,
            hidden_segments=cfg.mlp.hidden.segments,
        )
        self.root_dir = f"{hydra.utils.get_original_cwd()}"
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def setup(self, stage):

        full_path = [f"{self.root_dir}/{path}" for path in self.cfg.filenames]
        self.train_dataset = SingleTextDataset(
            filenames=full_path, max_size=self.cfg.data.max_size)
        self.test_dataset = SingleTextDataset(
            filenames=full_path, max_size=self.cfg.data.max_size)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.flatten())

        self.log(f'train_loss', loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

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


@hydra.main(config_path="./config", config_name="language_config")
def run_language_interpolation(cfg: DictConfig):
    # TODO use a space filling curve to map x,y linear coordinates
    # to space filling coordinates 1d coordinate.
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    if cfg.train is True:
        trainer = Trainer(max_epochs=cfg.max_epochs, gpus=cfg.gpus)
        model = Net(cfg)
        trainer.fit(model)
        print('testing')
        result = trainer.test(model)
        print('result', result)
        print('finished testing')
        print('best check_point', trainer.checkpoint_callback.best_model_path)
        print('loss', result[0]['train_loss'])
        return result[0]['train_loss']
    else:
        # plot some data
        print('evaluating result')
        print('cfg.checkpoint', cfg.checkpoint)
        checkpoint_path = f"{hydra.utils.get_original_cwd()}/{cfg.checkpoint}"
        print('checkpoint_path', checkpoint_path)
        model = Net.load_from_checkpoint(checkpoint_path)
        model.eval()
        text_in = cfg.text
        print('prompt:', text_in)
        
        for i in range(cfg.num_predict) :
            encoding, text_used = encode_input_from_text(text_in=text_in, features=10)
            encoding = ascii_to_float(encoding).unsqueeze(dim=0)
            model.eval()
            output = model(encoding)
            values, indices, ascii = decode_output_to_text(encoding=output[0], topk=cfg.topk)
            
            # pick the next character weighted by probabilities of each character
            # prevents the same response for every query.
            actual = random.choices(ascii, values.tolist())
            text_in = text_in+actual[0]
            
        print('output:', text_in.replace('\n',' '))


if __name__ == "__main__":
    run_language_interpolation()
