# from PIL import Image
from matplotlib import image
import torch
from torch import Tensor
import numpy as np
import math
import logging
from torch.utils.data import DataLoader, Dataset
from typing import Optional, List
from pytorch_lightning import LightningDataModule
from PIL import Image
from torchvision import transforms

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

    torch_image_flat = torch_image.reshape(-1, 3) * 2.0 / 255.0 - 1

    return torch_image_flat, torch_position, torch_image


def image_neighborhood_dataset(
    image: Tensor, width: int = 3, outside: int = 1, stride: int = 1
):
    """
    Args :
        image : Normalized image tensor in range [-1 to 1]
        width: width of the inner block.
        outside : width of the outer neighborhood surrounding the inner block.
    Return :
        tensor of inner block, tensor of neighborhood
    """

    px = image.shape[1]
    py = image.shape[2]

    max_x = px - (width + 2 * outside)
    max_y = py - (width + 2 * outside)
    lastx = max_x
    lasty = max_y

    totalx = width + 2 * outside
    totaly = totalx

    edge_mask = torch.ones(totalx, totaly, dtype=bool)
    edge_mask[outside : (outside + width), outside : (outside + width)] = False
    block_mask = ~edge_mask

    edge_indexes = edge_mask.flatten()
    block_indexes = block_mask.flatten()

    image = image.unsqueeze(0)
    patches = (
        image.unfold(2, totalx, stride)
        .unfold(3, totaly, stride)
        .flatten(start_dim=4, end_dim=5)
    )

    patches = patches.squeeze(0).permute(1, 2, 0, 3)

    patch_block = patches[:, :, :, block_indexes].flatten(2)
    patch_edge = patches[:, :, :, edge_indexes].flatten(2)

    patch_block = patch_block.reshape(patch_block.shape[0] * patch_block.shape[1], -1)
    patch_edge = patch_edge.reshape(patch_edge.shape[0] * patch_edge.shape[1], -1)

    return patch_edge, patch_block, image.squeeze(0), lastx, lasty


class ImageNeighborhoodReader:
    def __init__(self, filename: str, width=3, outside=1):

        img = Image.open(filename)
        image = transforms.ToTensor()(img) * 2 - 1

        (
            self._input,
            self._output,
            self._image,
            self._lastx,
            self._lasty,
        ) = image_neighborhood_dataset(image=image, width=width, outside=outside)

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


class ImageNeighborhoodDataset(Dataset):
    def __init__(self, filenames: List[str], width: int = 3, outside: int = 1):
        # TODO: right now grabbing the first element
        ind = ImageNeighborhoodReader(filenames[0], width=width, outside=outside)
        self.inputs = ind.features
        self.output = ind.targets
        self.image = ind.image
        self.lastx = ind.lastx
        self.lasty = ind.lasty
        self.image_neighborhood = ind

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.inputs[idx], self.output[idx]


class ImageNeighborhoodDataModule(LightningDataModule):
    def __init__(
        self,
        filenames: List[str] = None,
        width: int = 3,
        outside: int = 1,
        num_workers: int = 10,
        pin_memory: int = True,
        batch_size: int = 32,
        shuffle=True,
    ):
        super().__init__()
        self._filenames = filenames
        self._width = width
        self._outside = outside
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._batch_size = batch_size
        self._shuffle = shuffle

    def setup(self, stage: Optional[str] = None):

        self._train_dataset = ImageNeighborhoodDataset(filenames=self._filenames)

        # Since I'm doing memorization, I actually want test and train to be the same
        self._test_dataset = ImageNeighborhoodDataset(filenames=self._filenames)

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            drop_last=True,  # Needed for batchnorm
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            drop_last=True,
        )


class ImageDataset(Dataset):
    def __init__(self, filenames: List[str], rotations: int = 1):
        self.output, self.input, self.image = image_to_dataset(
            filenames[0], rotations=rotations
        )

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.input[idx], self.output[idx]


class ImageDataModule(LightningDataModule):
    def __init__(
        self,
        filenames: List[str],
        num_workers: int = 10,
        pin_memory: int = True,
        batch_size: int = 32,
        shuffle: bool = True,
        rotations: int = 2,
    ):
        super().__init__()
        self._filenames = filenames
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._rotations = rotations

    def setup(self, stage: Optional[str] = None):

        self._train_dataset = ImageDataset(
            filenames=self._filenames, rotations=self._rotations
        )
        self._test_dataset = ImageDataset(
            filenames=self._filenames, rotations=self._rotations
        )

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            drop_last=True,  # Needed for batchnorm
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=self._pin_memory,
            num_workers=self._num_workers,
            drop_last=True,
        )
