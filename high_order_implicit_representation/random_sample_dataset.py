from typing import Optional
from pathlib import Path
from high_order_layers_torch.utils import positions_from_mesh
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import pytorch_lightning as pl
from torchvision import transforms
from PIL import Image
from typing import List, Tuple
import logging
from torch import Tensor
import math

logger = logging.getLogger(__name__)


def indices_from_grid(image_size: int, device):
    xv, yv = torch.meshgrid([torch.arange(image_size), torch.arange(image_size)])
    indices = torch.stack([xv, yv]).permute(1, 2, 0).reshape(-1, 2).to(device=device)
    linear_indices = indices[:, 0] + indices[:, 1] * image_size
    return indices, linear_indices


def standard_transforms(image_size: int, mean: float = 0.5, std: float = 0.5):
    """
    Transform and image and normalize while converting to pytorch tensor.
    """
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transform


class RandomImageSampleDataset(Dataset):
    def __init__(
        self,
        image_size: int,
        path_list: List[str],
        num_feature_pixels: int = 25,
        num_target_pixels: int = 1,
        rotations: int = 1,
        device="cpu",
    ):
        """
        Uniform random sample interpolation without overlap in either the
        sample points or interpolation points.  I expect this to be replaced
        by the radial sampler as this one produces too much noise in the results
        since there is no bias towards the sample point.
        """
        super().__init__()
        self._image_size = image_size
        self._paths = path_list
        self._num_feature_pixels = num_feature_pixels
        self._num_target_pixels = num_target_pixels

        self._transform = standard_transforms(image_size=image_size)

        self._stripe_list = positions_from_mesh(
            width=image_size,
            height=image_size,
            device=device,
            rotations=rotations,
            normalize=True,
        )

        self._feature_fraction = num_feature_pixels / (
            num_target_pixels + num_feature_pixels
        )

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        path = self._paths[index]
        img = Image.open(path)

        img = self._transform(img)

        new_vals = torch.cat([val.unsqueeze(0) for val in self._stripe_list])

        # Just stack the x, y... positions as additional channels
        ans = torch.cat([img, new_vals])

        c, h, w = ans.shape
        ans = ans.reshape(c, -1).permute(1, 0)
        size = ans.shape[0]
        channels = ans.shape[1]

        feature_size = int(self._feature_fraction * size)
        feature_examples = feature_size // self._num_feature_pixels
        target_examples = (size - feature_size) // self._num_target_pixels

        examples = min(feature_examples, target_examples)

        indices = torch.randperm(size)

        # corrected size
        feature_size = examples * self._num_feature_pixels
        target_size = examples * self._num_target_pixels

        feature_indices = indices[:feature_size]
        target_indices = indices[feature_size : (feature_size + target_size)]

        features = ans[feature_indices, :].reshape(
            examples, self._num_feature_pixels, channels
        )
        targets = ans[target_indices, :].reshape(
            examples, self._num_target_pixels, channels
        )

        # We want all positions to be measured from the target and then
        # we only want to predict the RGB component of the target, so
        # This assumes these are RGB (3 channel images)
        features[:, :, 3:] = features[:, :, 3:] - targets[:, :, 3:]

        return features, targets[:, :, :3]  # only return RGB of target


class RadialRandomImageSampleDataset(Dataset):
    def __init__(
        self,
        image_size: int,
        path_list: List[str],
        num_feature_pixels: int = 25,
        num_target_pixels: int = 1,
        rotations: int = 1,
        device: str = "cpu",
    ):
        """
        Sampling data along r, theta instead of uniform x,y
        """
        super().__init__()
        self._image_size = image_size
        self._paths = path_list
        self._num_feature_pixels = num_feature_pixels
        self._num_target_pixels = num_target_pixels

        self._transform = standard_transforms(image_size=image_size)

        self._stripe_list = positions_from_mesh(
            width=image_size,
            height=image_size,
            device=device,
            rotations=rotations,
            normalize=True,
        )
        self._stripe_list = torch.cat([val.unsqueeze(0) for val in self._stripe_list])

        self._indices, self._target_linear_indices = indices_from_grid(
            image_size, device=device
        )

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        path = self._paths[index]
        img = Image.open(path)
        img = self._transform(img)
        return random_radial_samples_from_image(
            img=img,
            stripe_list=self._stripe_list,
            image_size=self._image_size,
            feature_pixels=self._num_feature_pixels,
            indices=self._indices,
            target_linear_indices=self._target_linear_indices,
        )


def random_radial_samples_from_image(
    img: Tensor,
    stripe_list: Tensor,
    image_size: int,
    feature_pixels: int,
    indices: Tensor,
    target_linear_indices: Tensor,
    device="cpu",
):
    """
    Given an image, produce a list of random interpolations for every pixel
    in the image. The image is reflected at all boundaries and interpolations
    that go beyond the image ranges use the reflected values.
    Args :
        indices : 2d sample indexes [[0,0],[0,1],...]
        target_linear_indices : flattened indexes [0...image_size*image_size-1]
    """

    # Just stack the x, y... positions as additional channels
    ans = torch.cat([img, stripe_list])

    c, h, w = ans.shape
    ans = ans.reshape(c, -1).permute(1, 0)

    ij_indices = random_symmetric_sample(
        image_size=image_size,
        interp_size=feature_pixels,
        samples=indices,
        device=device,
    )

    feature_linear_indices = ij_indices[:, :, 0] + ij_indices[:, :, 1] * image_size
    features = ans[feature_linear_indices, :]
    targets = ans[target_linear_indices, :]
    targets = targets.reshape(targets.shape[0], 1, targets.shape[1])

    # We want all positions to be measured from the target and then
    # we only want to predict the RGB component of the target, so
    # This assumes these are RGB (3 channel images)
    # The maximum diff here should be +-0.5
    features[:, :, 3:] = features[:, :, 3:] - targets[:, :, 3:]

    return features, targets[:, :, :3]  # only return RGB of target


def random_symmetric_sample(
    image_size: int, interp_size: int, samples: Tensor, device="cpu"
) -> Tensor:
    """
    Create sample points that are computed with random r and theta.  This naturally
    leads to a distribution that is lower density as the radius increases.
    Args :
        image_size : Assumed square, width of image
        sample_size : number of interpolation points
        sample : number of samples
    Returns :
        i,j grid indexes to use to use for each sample [num_samples, sample_size, 2]
    """
    num_samples = samples.shape[0]

    # r, theta
    r = torch.rand(num_samples, interp_size, device=device)
    theta = torch.rand(num_samples, interp_size, device=device) * 2 * math.pi

    # a pixel at the center should sample the edge but
    # not further.
    x = (0.5 * r * torch.cos(theta) * image_size).int()
    y = (0.5 * r * torch.sin(theta) * image_size).int()

    xy = torch.stack([x, y]).permute(2, 1, 0)
    xy = xy + samples

    # reflect all values that fall outside the boundary
    xy = torch.where(xy > (image_size - 1), 2 * (image_size - 1) - xy, xy)
    xy = torch.where(xy < 0, -xy, xy)
    xy = xy.permute(1, 0, 2)

    return xy


def random_image_sample_collate_fn(batch):
    feature_set = [features for features, targets in batch]
    target_set = [targets for features, targets in batch]

    features = torch.cat(feature_set)
    targets = torch.cat(target_set)

    return features, targets


class RandomImageSampleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_size: int,
        folder: str,
        num_feature_pixels: int,
        num_target_pixels: int,
        exts: List[str] = ["jpg", "jpeg", "png", "JPEG"],
        batch_size: int = 32,
        num_workers: int = 10,
        split_frac: float = 0.8,
        root_dir: str = ".",
        rotations: int = 1,
        dataset: Dataset = RadialRandomImageSampleDataset,
    ):
        super().__init__()
        self._image_size = image_size
        self._folder = folder
        self._num_feature_pixels = num_feature_pixels
        self._num_target_pixels = num_target_pixels
        self._exts = exts
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._split_frac = split_frac
        self._root_dir = root_dir
        self._rotations = rotations
        self._dataset = dataset

    def setup(self, stage: Optional[str] = None):

        folder = (Path(self._root_dir) / self._folder).as_posix()

        if Path(folder).exists() is False:
            raise ValueError(f"Folder {folder} does not exist.")

        paths = [
            p.as_posix()
            for ext in self._exts
            for p in Path(f"{folder}").glob(f"**/*.{ext}")
        ]
        logger.info(f"Image paths [:10] in folder '{self._folder}' are {paths[:10]}")
        size = len(paths)

        train_size = int(self._split_frac * size)
        test_size = (size - train_size) // 2
        val_size = size - train_size - test_size

        self._train_list, self._test_list, self._val_list = [
            list(val) for val in random_split(paths, [train_size, test_size, val_size])
        ]

        self._train_dataset = self._dataset(
            image_size=self._image_size,
            path_list=self._train_list,
            num_feature_pixels=self._num_feature_pixels,
            num_target_pixels=self._num_target_pixels,
            rotations=self._rotations,
        )

        self._val_dataset = self._dataset(
            image_size=self._image_size,
            path_list=self._val_list,
            num_feature_pixels=self._num_feature_pixels,
            num_target_pixels=self._num_target_pixels,
            rotations=self._rotations,
        )
        self._test_dataset = self._dataset(
            image_size=self._image_size,
            path_list=self._test_list,
            num_feature_pixels=self._num_feature_pixels,
            num_target_pixels=self._num_target_pixels,
            rotations=self._rotations,
        )

        logger.info(f"Train dataset size is {len(self._train_dataset)}")
        logger.info(f"Validation dataset size is {len(self._val_dataset)}")
        logger.info(f"Test dataset size is {len(self._test_dataset)}")

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    @property
    def val_dataset(self):
        return self._val_dataset

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self._num_workers,
            collate_fn=random_image_sample_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self._num_workers,
            collate_fn=random_image_sample_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self._num_workers,
            collate_fn=random_image_sample_collate_fn,
        )
