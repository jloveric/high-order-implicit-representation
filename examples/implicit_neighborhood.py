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
)
from high_order_implicit_representation.rendering import (
    NeighborGenerator,
    generate_sequence,
)
from pytorch_lightning.callbacks import LearningRateMonitor
from high_order_layers_torch.networks import *
import logging
from torchvision.utils import make_grid
from PIL import Image
from torchvision import transforms

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

        # Choose an output size that evenly splits the full block width+2*outside
        # So as not to see artifacts from the boundary.
        image_generator = NeighborGenerator(
            samples=5,
            frames=25,
            output_size=[180, 180],
            width=3,
            outside=3,
            batch_size=cfg.batch_size // 8,
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
        logger.info(f"cfg.checkpoint {cfg.checkpoint}")
        checkpoint_path = f"{hydra.utils.get_original_cwd()}/{cfg.checkpoint}"

        logger.info(f"checkpoint_path {checkpoint_path}")
        model = Net.load_from_checkpoint(checkpoint_path)
        model.eval()
        image_dir = [f"{hydra.utils.get_original_cwd()}/{f}" for f in cfg.images]

        logger.info(f"image_dir {image_dir}")

        img = Image.open(image_dir[0])
        image = transforms.ToTensor()(img) * 2 - 1

        all_images = generate_sequence(
            model=model,
            image=image,
            frames=1,
            width=3,
            outside=3,
            batch_size=256,
        )

        img = make_grid(all_images).permute(1, 2, 0).cpu().numpy()

        plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    run_implicit_neighborhood()
