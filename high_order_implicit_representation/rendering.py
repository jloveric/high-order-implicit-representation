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
from high_order_implicit_representation.single_image_dataset import (
    image_neighborhood_dataset,
)
import math

logger = logging.getLogger(__name__)
default_size = [64, 64]


def neighborhood_sample_generator(
    model: nn.Module,
    image: Tensor,
    width: int,
    outside: int,
):
    """
    Create a bunch of samples with size width x width and and stride
    width computed from the input values.  The input values are a (square) donut
    around the the central block.
    Args :
      model : the model
      image : image tensor in the form [C, H, W] and assumed normalized to [-1, 1]
      width : width of the central block w x w
      outside : width of surrounding cells.  Total block has size (width+2*outside)^2 so
      the out side material is the feature set.
    Returns :
      A new image with updated values assuming 2d reflection boundary conditions
    """

    # Adding extra padding to the boundary to account for edge cases.
    tpad = 2 * outside + width

    ext_image = nn.ReflectionPad2d(padding=tpad)(image)

    output = image_neighborhood_dataset(
        image=ext_image,
        width=width,
        outside=outside,
        stride=width,
    )
    features = output[0]

    total_size = width + 2 * outside

    imax, jmax = ext_image.shape[1:3]

    # Extents for starting new block of size total_size
    ni = imax - total_size + 1
    nj = jmax - total_size + 1

    targets = model(features)
    targets = targets.permute(1, 0)
    targets = targets.unsqueeze(0)  # one batch

    reduced_size = [math.ceil(ni / width) * width, math.ceil(nj / width) * width]

    # convert targets back into image
    result = nn.functional.fold(
        input=targets,
        output_size=reduced_size,
        kernel_size=width,
        stride=width,
    )

    """
    dshape = (torch.tensor(image.shape) - torch.tensor(result.shape[1:])) // 2

    dshape = dshape[1:]

    # It's crazy, but padding is reversed from typical order H, W and list below as W, H
    padding = (
        dshape[1],
        dshape[1],
        dshape[0],
        dshape[0],
    )

    # pad targets
    #this_image = nn.ReflectionPad2d(padding=padding)(result)
    """
    return result.squeeze(0)[
        :, tpad : (tpad + image.shape[1]), tpad : (tpad + image.shape[2])
    ]


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
            this_image = (
                torch.rand(3, self._output_size[0], self._output_size[1]) * 2 - 1
            )
            for f in range(self._frames):
                this_image = neighborhood_sample_generator(
                    model=pl_module,
                    image=this_image,
                    outside=self._outside,
                )
                images_list.append(this_image)

            all_images = torch.cat(images_list, dim=0)

            img = make_grid(all_images).permute(1, 2, 0).cpu().numpy()

            # PLOT IMAGES
            trainer.logger.experiment.add_image(
                f"sample {e}",
                torch.tensor(img).permute(2, 0, 1),
                global_step=trainer.global_step,
            )
