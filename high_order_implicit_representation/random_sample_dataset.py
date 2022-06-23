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
        super().__init__()
        self._image_size = image_size
        self._paths = path_list
        self._num_feature_pixels = num_feature_pixels
        self._num_target_pixels = num_target_pixels

        self._transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

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


def make_periodic(x, periodicity: float):
    xp = x + 0.5 * periodicity
    xp = torch.remainder(xp, 2 * periodicity)  # always positive
    xp = torch.where(xp > periodicity, 2 * periodicity - xp, xp)
    xp = xp - 0.5 * periodicity
    return xp


def random_symmetric_sample(
    image_size: int, interp_size: int, samples: Tensor
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
    r = torch.rand(num_samples, interp_size)
    theta = torch.rand(num_samples, interp_size) * 2 * math.pi

    x = (r * torch.cos(theta) * image_size).int()
    y = (r * torch.sin(theta) * image_size).int()

    print('x.shape', x.shape, y.shape)
    xy = torch.stack([x, y]).permute(2,1,0)
    print('xy.shape', xy.shape, 'samples.shape', samples.shape)
    xy = xy + samples

    # reflect all values that fall outside the boundary
    xy = torch.where(xy > (image_size-1), 2 * (image_size-1) - xy, xy)
    xy = torch.where(xy < 0, -xy, xy)
    xy = xy.permute(1,0,2)

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

        self._train_dataset = RandomImageSampleDataset(
            image_size=self._image_size,
            path_list=self._train_list,
            num_feature_pixels=self._num_feature_pixels,
            num_target_pixels=self._num_target_pixels,
            rotations=self._rotations,
        )

        self._val_dataset = RandomImageSampleDataset(
            image_size=self._image_size,
            path_list=self._val_list,
            num_feature_pixels=self._num_feature_pixels,
            num_target_pixels=self._num_target_pixels,
            rotations=self._rotations,
        )
        self._test_dataset = RandomImageSampleDataset(
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
