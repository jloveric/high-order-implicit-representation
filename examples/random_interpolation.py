from typing import List
from pathlib import Path
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
    standard_transforms,
)
from pytorch_lightning.loggers import TensorBoardLogger
import torch_optimizer as alt_optim
from high_order_implicit_representation.utils import (
    ImageSampler,
    generate_sample,
    generate_sample_radial,
)
from PIL import Image

# from high_order_mlp import HighOrderMLP
from high_order_implicit_representation.single_image_dataset import image_to_dataset
from torch.utils.data import DataLoader, Dataset
import logging
from high_order_implicit_representation.networks import Net
import matplotlib

logging.basicConfig()
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)


@hydra.main(config_path="../config", config_name="random_interpolation")
def run_implicit_images(cfg: DictConfig):

    root_dir = hydra.utils.get_original_cwd()

    logger.info(OmegaConf.to_yaml(cfg))
    logger.info(f"Working directory : {os.getcwd()}")
    logger.info(f"Orig working directory    : {root_dir}")

    if cfg.train is True:

        image_sampler = ImageSampler(
            features=cfg.num_feature_pixels,
            targets=cfg.num_target_pixels,
            rotations=cfg.rotations,
            image_size=cfg.image_size,
        )

        datamodule = RandomImageSampleDataModule(
            image_size=cfg.image_size,
            folder=cfg.folder,
            num_target_pixels=cfg.num_target_pixels,
            num_feature_pixels=cfg.num_feature_pixels,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            split_frac=cfg.split_frac,
            root_dir=root_dir,
            rotations=cfg.rotations,
        )
        tb_logger = TensorBoardLogger("tb_logs", name="diffusion")
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        early_stopping = EarlyStopping(monitor="train_loss", patience=cfg.patience)
        trainer = Trainer(
            callbacks=[early_stopping, lr_monitor, image_sampler],
            max_epochs=cfg.max_epochs,
            gpus=cfg.gpus,
            logger=tb_logger,
        )

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
        checkpoint_path = f"{root_dir}/{cfg.checkpoint}"
        logger.info("checkpoint_path {checkpoint_path}")
        model = Net.load_from_checkpoint(checkpoint_path)

        img = None
        if cfg.image is not None:
            img = Image.open(f"{root_dir}/{cfg.image}")
            img = standard_transforms(image_size=cfg.image_size)(img)

        image_samples = generate_sample_radial(
            model=model,
            features=cfg.num_feature_pixels,
            targets=cfg.num_target_pixels,
            rotations=cfg.rotations,
            image_size=cfg.image_size,
            iterations=cfg.iterations,
            all_random=cfg.all_random,
            start_image=img,
        )

        image_samples = [0.5 * (image.permute(1, 2, 0) + 1) for image in image_samples]
        print(image_samples[0])

        size = len(image_samples)
        width = int(math.sqrt(size))
        height = width
        if width * width < len(image_samples):
            height += 1

        f = plt.figure(figsize=(4, 4))  # Notice the equal aspect ratio
        axarr = [f.add_subplot(width, height, i + 1) for i in range(len(image_samples))]
        # f, axarr = plt.subplots(width, height, gridspec_kw={"wspace": 0, "hspace": 0})
        # axarr = axarr.flatten()
        for index, sample in enumerate(image_samples):
            axarr[index].imshow(sample.detach().numpy(), interpolation="none")
            axarr[index].axis("off")
            axarr[index].set_aspect("equal")

        f.subplots_adjust(wspace=0, hspace=0)

        fname = (Path(root_dir) / "results" / "random_interpolation.png").as_posix()
        if cfg.save_plots:
            plt.savefig(
                fname,
                dpi="figure",
            )

        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    run_implicit_images()
