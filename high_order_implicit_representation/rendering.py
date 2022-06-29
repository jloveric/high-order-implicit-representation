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
    image_to_dataset,
)
import math
import matplotlib.pyplot as plt
import io
import PIL
from torchvision import transforms

logger = logging.getLogger(__name__)
default_size = [64, 64]


def neighborhood_sample_generator(
    model: nn.Module,
    image: Tensor,
    width: int,
    outside: int,
    batch_size: int = 256,
    device: str = "cpu",
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
      device: The device things are on.
    Returns :
      A new image with updated values assuming 2d reflection boundary conditions
    """

    # Adding extra padding to the boundary to account for edge cases.
    tpad = 2 * outside + width

    ext_image = nn.ReflectionPad2d(padding=tpad)(image)

    output = image_neighborhood_dataset(
        image=ext_image, width=width, outside=outside, stride=width, device=device
    )
    features = output[0]

    total_size = width + 2 * outside

    imax, jmax = ext_image.shape[1:3]

    # Extents for starting new block of size total_size
    ni = imax - total_size + 1
    nj = jmax - total_size + 1

    target_list = []
    for batch in range((len(features) + batch_size) // batch_size):
        target_list.append(
            model(features[batch * batch_size : (batch + 1) * batch_size])
        )
    targets = torch.cat(target_list)

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

    return result.squeeze(0)[
        :, tpad : (tpad + image.shape[1]), tpad : (tpad + image.shape[2])
    ]


class NeighborGenerator(Callback):
    def __init__(
        self,
        samples: int = 5,
        frames: int = 10,
        output_size: List[int] = default_size,
        width=3,
        outside=1,
        batch_size: int = 256,
    ) -> None:
        super().__init__()
        self._samples = samples
        self._width = width
        self._outside = outside
        self._output_size = output_size
        self._frames = frames
        self._batch_size = batch_size

    @rank_zero_only
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.eval()
        with torch.no_grad():
            for e in range(self._samples):
                images_list = []
                this_image = (
                    torch.rand(
                        3,
                        self._output_size[0],
                        self._output_size[1],
                        device=pl_module.device,
                    )
                    * 2
                    - 1
                )
                for _ in range(self._frames):
                    new_image = neighborhood_sample_generator(
                        model=pl_module,
                        image=this_image,
                        width=self._width,
                        outside=self._outside,
                        device=pl_module.device,
                        batch_size=self._batch_size,
                    )
                    this_image = 0.5 * (this_image + new_image)
                    images_list.append(this_image.cpu().clone().detach())

                all_images = torch.stack(images_list, dim=0).detach()
                all_images = 0.5 * (all_images + 1)

                img = make_grid(all_images).permute(1, 2, 0).cpu().numpy()

                trainer.logger.experiment.add_image(
                    f"sample {e}",
                    torch.tensor(img).permute(2, 0, 1),
                    global_step=trainer.global_step,
                )


"""
EXAMPLE FROM PDE
def generate_images(model: nn.Module, save_to: str = None, layer_type: str = None):

    model.eval()
    inputs = pde_grid().detach().to(model.device)
    y_hat = model(inputs).detach().cpu().numpy()
    outputs = y_hat.reshape(100, 100, 3)

    names = ["Density", "Velocity", "Pressure"]

    image_list = []
    for j, name in enumerate(names):

        plt.figure(j + 1)
        fig, (ax0, ax1) = plt.subplots(2, 1)

        # The outputs are density, momentum and energy
        # so each of the components 0, 1, 2 represents
        # on of those quantities
        c = ax0.pcolor(outputs[:, :, j])
        ax0.set_xlabel("x")
        ax0.set_ylabel("time")

        for i in range(0, 100, 20):
            d = ax1.plot(outputs[:, i, j], label=f"t={i}")

            ax1.set_xlabel("x")
            ax1.set_ylabel(f"{name}")

        ax1.legend()

        ax0.set_title(f"{name} with {layer_type} layers")
        plt.xlabel("x")

        if save_to == "file":
            this_path = f"{hydra.utils.get_original_cwd()}"
            plt.savefig(
                f"{this_path}/images/{name}-{layer_type}",
                dpi="figure",
                format=None,
                metadata=None,
                bbox_inches=None,
                pad_inches=0.1,
                facecolor="auto",
                edgecolor="auto",
                backend=None,
            )
        elif save_to == "memory":
            buf = io.BytesIO()
            plt.savefig(
                buf,
                dpi="figure",
                format=None,
                metadata=None,
                bbox_inches=None,
                pad_inches=0.1,
                facecolor="auto",
                edgecolor="auto",
                backend=None,
            )
            buf.seek(0)
            image = PIL.Image.open(buf)
            image = transforms.ToTensor()(image)
            image_list.append(image)

    if save_to != "memory":
        plt.show()
    else:
        return image_list

    return None
"""


class ImageGenerator(Callback):
    def __init__(self, filename, rotations, batch_size):
        _, self._inputs, self._image = image_to_dataset(filename, rotations=rotations)
        self._batch_size = batch_size

    @rank_zero_only
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.eval()
        with torch.no_grad():
            self._inputs = self._inputs.to(device=pl_module.device)

            y_hat_list = []
            batches = (len(self._inputs) + self._batch_size) // self._batch_size
            for batch in range(batches):
                res = pl_module(
                    self._inputs[
                        batch * self._batch_size : (batch + 1) * self._batch_size
                    ]
                )
                y_hat_list.append(res.detach().cpu())
            y_hat = torch.cat(y_hat_list)

            ans = y_hat.reshape(
                self._image.shape[0], self._image.shape[1], self._image.shape[2]
            )
            ans = 0.5 * (ans + 1.0)

            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(ans.detach().cpu().numpy())
            axarr[0].set_title("fit")
            axarr[1].imshow(self._image.cpu())
            axarr[1].set_title("original")

            for i in range(2):
                axarr[i].axes.get_xaxis().set_visible(False)
                axarr[i].axes.get_yaxis().set_visible(False)

            buf = io.BytesIO()
            plt.savefig(
                buf,
                dpi="figure",
                format=None,
                metadata=None,
                bbox_inches=None,
                pad_inches=0.1,
                facecolor="auto",
                edgecolor="auto",
                backend=None,
            )
            buf.seek(0)
            image = PIL.Image.open(buf)
            image = transforms.ToTensor()(image)

            trainer.logger.experiment.add_image(
                f"image", image, global_step=trainer.global_step
            )
