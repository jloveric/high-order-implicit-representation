# from PIL import Image
from matplotlib import image
import torch
from torch import Tensor
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
import math
import logging

logger = logging.getLogger(__name__)


def image_to_dataset(filename: str, peano: str = False, rotations: int = 1):
    """
    Read in an image file and return the flattened position input
    flattened output and torch array of the original image.def image_to_dataset(filename: str, peano: str = False, rotations: int = 1):

    Args :
        filename : image filename.
    Returns :
        flattened image [width*heigh, rgb], flattened position vectory
        [width*height, 2] and torch tensor of original image.
    """
    img = image.imread(filename)

    torch_image = torch.from_numpy(np.array(img))
    max_size = max(torch_image.shape[0], torch_image.shape[1])

    xv, yv = torch.meshgrid(
        [torch.arange(torch_image.shape[0]), torch.arange(torch_image.shape[1])]
    )

    # rescale so the maximum values is between -1 and 1
    xv = (xv / max_size) * 2 - 1
    yv = (yv / max_size) * 2 - 1

    xv = xv.reshape(xv.shape[0], xv.shape[1], 1)
    yv = yv.reshape(yv.shape[0], yv.shape[1], 1)

    """
    if peano is True:
        # can index 2^{n*p} cubes with p = 2 (dimension)
        n = 2  # number of dimensions
        p = math.ceil(math.log(max_size, n)/2.0)
        hilbert_curve = HilbertCurve(p=p, n=2)
        cartesian_position = torch_position.tolist()
        hilbert_distances = hilbert_curve.distance_from_points(
            cartesian_position)
    """

    if rotations == 2:
        torch_position = torch.cat([xv, yv, (xv - yv) / 2.0, (xv + yv) / 2.0], dim=2)
        torch_position = torch_position.reshape(-1, 4)
    elif rotations == 1:
        torch_position = torch.cat([xv, yv], dim=2)
        torch_position = torch_position.reshape(-1, 2)
    else:
        line_list = []
        for i in range(rotations):
            theta = (math.pi / 2.0) * (i / rotations)
            rot_x = math.cos(theta)
            rot_y = math.sin(theta)
            rot_sum = math.fabs(rot_x) + math.fabs(rot_y)

            # Add the line and the line orthogonal
            line_list.append((rot_x * xv + rot_y * yv) / rot_sum)
            line_list.append((rot_x * xv - rot_y * yv) / rot_sum)

        torch_position = torch.cat(line_list, dim=2)
        torch_position = torch_position.reshape(-1, 2 * rotations)

        # raise(f"Rotation {rotations} not implemented.")

    torch_image_flat = torch_image.reshape(-1, 3) * 2.0 / 255.0 - 1

    return torch_image_flat, torch_position, torch_image


class ImageNeighborhoodReader:
    def __init__(self, filename: str, width=3, outside=1):
        self._input, self._output, self._image = self.image_neighborhood_dataset(
            filename=filename, width=width, outside=outside
        )

    @property
    def features(self) -> Tensor:
        return self._input

    @property
    def targets(self) -> Tensor:
        return self._output

    @property
    def image(self) -> Tensor:
        return self._image

    @property
    def lastx(self) -> int:
        return self._lastx

    @property
    def lasty(self) -> int:
        return self._lasty

    def image_neighborhood_dataset(self, filename: str, width=3, outside=1):
        """
        Args :
            filename : Name of image file to create data from.
            width: width of the inner block.
            outside : width of the outer neighborhood surrounding the inner block.
        Return :
            tensor of inner block, tensor of neighborhood
        """
        img = image.imread(filename)

        torch_image = torch.from_numpy(np.array(img))

        px = torch_image.shape[0]
        py = torch_image.shape[1]

        patch_edge = []
        patch_block = []

        max_x = px - (width + 2 * outside)
        max_y = py - (width + 2 * outside)
        self._lastx = max_x
        self._lasty = max_y

        totalx = width + 2 * outside
        totaly = totalx

        edge_mask = torch.ones(totalx, totaly, 3, dtype=bool)
        edge_mask[outside : (outside + width), outside : (outside + width), :] = False
        block_mask = ~edge_mask

        edge_indexes = edge_mask.flatten()
        block_indexes = block_mask.flatten()

        for i in range(max_x):
            for j in range(max_y):
                all_elements = torch_image[
                    i : (i + totalx), j : (j + totaly), :
                ].flatten()

                patch = all_elements[block_indexes]
                edge = all_elements[edge_indexes]

                patch_edge.append(edge)
                patch_block.append(patch)

        patch_block = (2.0 / 255.0) * torch.stack(patch_block) - 1
        patch_edge = (2.0 / 255.0) * torch.stack(patch_edge) - 1

        return patch_block, patch_edge, torch_image
