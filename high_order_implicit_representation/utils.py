import torch
from torch import Tensor
import logging
from stripe_layers.StripeLayer import create_stripe_list
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def generate_sample(
    model: torch.nn,
    features: int = 10,
    targets: int = 1,
    iterations: int = 10,
    start: Tensor = None,
    root_dir: str = None,
    image_size: int = 64,
    return_intermediate: bool = True,
    device: str = "cpu",
    rotations: int = 2,
    batch_size: int = 256,
):

    num_pixels = image_size * image_size
    model.eval()

    image = torch.rand([3, image_size, image_size]) * 2 - 1
    stripe_list = create_stripe_list(
        width=image_size, height=image_size, device=device, rotations=rotations
    )
    new_vals = torch.cat([val.unsqueeze(0) for val in stripe_list]) / (0.5 * image_size)

    result_list = []
    for count in range(iterations):

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
        features[:, :, 3:] = features[:, :, 3:] - targets[:, :, 3:]
        result = model(features)
        image = result.reshape(image_size, image_size, 3).permute(2, 0, 1)

        result_list.append(image)

    return result_list
