import torch
from torch import Tensor
import logging
from stripe_layers.StripeLayer import create_stripe_list
import matplotlib.pyplot as plt
from typing import List

logger = logging.getLogger(__name__)


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
        image = torch.rand([3, image_size, image_size]) * 2 - 1
    else:
        image = (image / 256) * 2 - 1

    stripe_list = create_stripe_list(
        width=image_size, height=image_size, device=device, rotations=rotations
    )
    new_vals = torch.cat([val.unsqueeze(0) for val in stripe_list]) / (0.5 * image_size)

    result_list = []
    for count in range(iterations):
        logger.info(f"Generating for count {count}")
        ans = torch.cat([image, new_vals])
        c, h, w = ans.shape
        ans = ans.reshape(c, -1).permute(1, 0)

        feature_indices = torch.remainder(
            torch.randperm(num_pixels * features), num_pixels
        )
        target_indices = torch.arange(0, num_pixels)

        features_tensor = ans[feature_indices].reshape(-1, features, c)
        targets_tensor = ans[target_indices].reshape(-1, 1, c)

        # Distances are measured relative to target
        features_tensor[:, :, 3:] = features_tensor[:, :, 3:] - targets_tensor[:, :, 3:]

        result = model(features_tensor.flatten(1))
        image = result.reshape(image_size, image_size, 3).permute(2, 0, 1)

        result_list.append(image)

    return result_list
