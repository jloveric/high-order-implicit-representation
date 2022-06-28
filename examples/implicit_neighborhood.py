import os
from omegaconf import DictConfig, OmegaConf
import hydra
from high_order_layers_torch.layers import *
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import torch
from high_order_implicit_representation.networks import Net
from high_order_implicit_representation.single_image_dataset import (
    ImageNeighborhoodDataModule,
    ImageNeighborhoodDataset,
)
from high_order_implicit_representation.rendering import NeighborGenerator
from pytorch_lightning.callbacks import LearningRateMonitor
from high_order_layers_torch.networks import *
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)


@hydra.main(config_path="../config", config_name="neighborhood_config")
def run_implicit_neighborhood(cfg: DictConfig):

    logger.info(OmegaConf.to_yaml(cfg))
    logger.info(f"Working directory {format(os.getcwd())}")
    logger.info(f"Orig working directory {hydra.utils.get_original_cwd()}")
    root_dir = hydra.utils.get_original_cwd()

    if cfg.train is True:
        full_path = [f"{root_dir}/{path}" for path in cfg.images]
        datamodule = ImageNeighborhoodDataModule(
            filenames=full_path, width=3, outside=3, batch_size=cfg.batch_size
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        image_generator = NeighborGenerator(
            samples=5, frames=10, output_size=[60, 60], width=3, outside=3
        )
        trainer = Trainer(
            max_epochs=cfg.max_epochs,
            gpus=cfg.gpus,
            callbacks=[lr_monitor, image_generator],
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
        checkpoint_path = f"{hydra.utils.get_original_cwd()}/{cfg.checkpoint}"

        print(f"checkpoint_path {checkpoint_path}")
        model = Net.load_from_checkpoint(checkpoint_path)
        model.eval()
        image_dir = f"{hydra.utils.get_original_cwd()}/{cfg.images[1]}"

        print(f"image_dir {image_dir}")
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
