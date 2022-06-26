import torch
from torch import Tensor
import logging
from high_order_layers_torch.utils import positions_from_mesh
import matplotlib.pyplot as plt
from typing import List
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torchvision.utils import make_grid
import numpy as np
from high_order_implicit_representation.random_sample_dataset import (
    random_radial_samples_from_image,
    indices_from_grid,
)
import copy

logger = logging.getLogger(__name__)


def generate_sample_radial(
    model: torch.nn,
    features: int = 10,
    targets: int = 1,
    iterations: int = 10,
    start_image: Tensor = None,
    image_size: int = 64,
    return_intermediate: bool = True,
    device: str = "cpu",
    rotations: int = 2,
    batch_size: int = 256,
    all_random: bool = True,
) -> List[Tensor]:
    """
    Create a sample either generated on top of a starting image
    or from a random image. Return a series of images with the
    for each time the network is applied.
    Args :
      model : The nn model
      features : The number of points used for interpolation
      targets : The number of target points where RGB will be predicted
      iterations : Number of times to apply the network to the image
      start_image : An normalized image to use as the initial value in range [-1, 1]
      image_size : The width of the image (assumed square)
      return_intermediate : Return intermediate values if true
      device : The device to perform operations on
      rotations : The number of rotations used by the network
      batch_size : Set the batch size to run through the network
    Returns :
      A list of images for each time the network is applied
    """

    num_pixels = image_size * image_size
    model.eval()

    if start_image is None:
        image = torch.rand([3, image_size, image_size], device=device) * 2 - 1
    else:
        image = copy.deepcopy(start_image)

    stripe_list = positions_from_mesh(
        width=image_size,
        height=image_size,
        device=device,
        rotations=rotations,
        normalize=True,
    )

    # These need to be normalized otherwise the max is the width of the grid
    stripe_list = torch.cat([val.unsqueeze(0) for val in stripe_list])

    indices, target_linear_indices = indices_from_grid(image_size, device=device)

    result_list = []
    for count in range(iterations):
        logger.info(f"Generating for count {count}")

        if all_random is True and start_image is None:
            image = torch.rand([3, image_size, image_size], device=device) * 2 - 1
        elif all_random is True and start_image is not None:
            image = copy.deepcopy(start_image)

        features_tensor, targets_tensor = random_radial_samples_from_image(
            img=image,
            stripe_list=stripe_list,
            image_size=image_size,
            feature_pixels=features,
            indices=indices,
            target_linear_indices=target_linear_indices,
            device=device,
        )

        result = model(features_tensor.flatten(1))
        image = result.reshape(image_size, image_size, 3).permute(2, 0, 1)

        result_list.append(image)

    return result_list


def generate_sample(
    model: torch.nn,
    features: int = 10,
    targets: int = 1,
    iterations: int = 10,
    image: Tensor = None,
    image_size: int = 64,
    return_intermediate: bool = True,
    device: str = "cpu",
    rotations: int = 2,
    batch_size: int = 256,
    all_random: bool = True,
) -> List[Tensor]:
    """
    Create a sample either generated on top of a starting image
    or from a random image. Return a series of images with the
    for each time the network is applied.
    Args :
      model : The nn model
      features : The number of points used for interpolation
      targets : The number of target points where RGB will be predicted
      iterations : Number of times to apply the network to the image
      image : An unnormalized image to use as the initial value
      image_size : The width of the image (assumed square)
      return_intermediate : Return intermediate values if true
      device : The device to perform operations on
      rotations : The number of rotations used by the network
      batch_size : Set the batch size to run through the network
    Returns :
      A list of images for each time the network is applied
    """

    num_pixels = image_size * image_size
    model.eval()

    if image is None:
        image = torch.rand([3, image_size, image_size], device=device) * 2 - 1
    else:
        image = (image / 255) * 2 - 1

    stripe_list = positions_from_mesh(
        width=image_size,
        height=image_size,
        device=device,
        rotations=rotations,
        normalize=True,
    )

    # These need to be normalized otherwise the max is the width of the grid
    new_vals = torch.cat([val.unsqueeze(0) for val in stripe_list])

    result_list = []
    for count in range(iterations):
        logger.info(f"Generating for count {count}")

        if all_random is True:
            image = torch.rand([3, image_size, image_size], device=device) * 2 - 1

        full_features = torch.cat([image, new_vals])
        channels, h, w = full_features.shape
        full_features = full_features.reshape(channels, -1).permute(1, 0)

        feature_indices = torch.remainder(
            torch.randperm(num_pixels * features, device=device), num_pixels
        )
        target_indices = torch.arange(start=0, end=num_pixels, device=device)

        features_tensor = full_features[feature_indices].reshape(-1, features, channels)
        targets_tensor = full_features[target_indices].reshape(-1, 1, channels)

        # Distances are measured relative to target so remove that component
        features_tensor[:, :, 3:] = features_tensor[:, :, 3:] - targets_tensor[:, :, 3:]

        result = model(features_tensor.flatten(1))
        image = result.reshape(image_size, image_size, 3).permute(2, 0, 1)

        result_list.append(image)

    return result_list


class ImageSampler(pl.callbacks.Callback):
    def __init__(
        self,
        batch_size: int = 36,
        samples: int = 36,
        features: int = 10,
        targets: int = 1,
        iterations: int = 10,
        image_size: int = 64,
        rotations: int = 2,
    ) -> None:
        super().__init__()
        self._batch_size = batch_size
        self._samples = samples
        self._features = features
        self._targets = targets
        self._iterations = iterations
        self._image_size = image_size
        self._rotations = rotations

    @rank_zero_only
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.eval()
        logger.info("Generating sample")
        all_images_list = generate_sample_radial(
            model=pl_module,
            features=self._features,
            targets=self._targets,
            iterations=self._iterations,
            image_size=self._image_size,
            rotations=self._rotations,
            batch_size=self._batch_size,
            device=pl_module.device,
        )

        all_images = torch.stack(all_images_list, dim=0).detach()
        all_images = 0.5 * (all_images + 1)

        img = make_grid(all_images).permute(1, 2, 0).cpu().numpy()

        trainer.logger.experiment.add_image(
            "img", torch.tensor(img).permute(2, 0, 1), global_step=trainer.global_step
        )
