from typing import List

import os
from omegaconf import DictConfig, OmegaConf
import hydra
from high_order_layers_torch.layers import *
from high_order_layers_torch.networks import *
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import torch
from high_order_implicit_representation.networks import Net

from high_order_implicit_representation.single_image_dataset import (
    image_to_dataset,
    ImageDataModule,
)
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)


@hydra.main(config_path="../config", config_name="images_config")
def run_implicit_images(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    root_dir = hydra.utils.get_original_cwd()

    if cfg.train is True:
        full_path = [f"{root_dir}/{path}" for path in cfg.images]
        data_module = ImageDataModule(
            filenames=full_path, batch_size=cfg.batch_size, rotations=cfg.rotations
        )
        trainer = Trainer(max_epochs=cfg.max_epochs, gpus=cfg.gpus)
        model = Net(cfg)
        trainer.fit(model, datamodule=data_module)
        print("testing")
        trainer.test(model, datamodule=data_module)
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
