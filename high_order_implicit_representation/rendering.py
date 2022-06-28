import torch
from typing import List
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import Callback
from torch import Tensor
from torchvision.utils import save_image
from torchvision.utils import make_grid
import logging
from torch import nn
from single_image_dataset import image_neighborhood_dataset

logger = logging.getLogger(__name__)
default_size = [64, 64]


class NeighborGenerator(Callback):
    def __init__(
        self,
        model: torch.nn.Module,
        samples: int = 5,
        frames: int = 10,
        output_size: List[int] = default_size,
        width=3,
        outside=1,
    ) -> None:
        super().__init__()
        self._model = model
        self._samples = samples
        self._width = width
        self._outside = outside
        self._output_size = output_size
        self._frames = frames

    @rank_zero_only
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.eval()

        for e in range(self._samples):
            images_list = []
            this_image = torch.rand(3, self._output_size, self._output_size) * 2 - 1
            for f in range(self._frames):
                output = image_neighborhood_dataset(
                    image=this_image, width=3, outside=3
                )
                features = output[0]
                targets = pl_module(features)

                # convert targets back into image

                # pad targets
                this_image = nn.ReflectionPad2d()
                images_list.append(this_image)

            all_images = torch.cat(images_list, dim=0)

            img = make_grid(all_images).permute(1, 2, 0).cpu().numpy()

            # PLOT IMAGES
            trainer.logger.experiment.add_image(
                f"sample {e}",
                torch.tensor(img).permute(2, 0, 1),
                global_step=trainer.global_step,
            )
