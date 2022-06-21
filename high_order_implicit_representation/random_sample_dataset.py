from typing import Optional
from pathlib import Path
from stripe_layers.StripeLayer import create_stripe_list
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import pytorch_lightning as pl
from torchvision import transforms
from PIL import Image
from typing import List
import logging

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

        self._stripe_list = create_stripe_list(
            width=image_size, height=image_size, device=device, rotations=rotations
        )

        self._feature_fraction = num_feature_pixels / (
            num_target_pixels + num_feature_pixels
        )

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, index):
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
        batch_size=32,
        num_workers=10,
        split_frac=0.8,
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

    def setup(self, stage: Optional[str] = None):

        self.folder = self._folder
        self.image_size = self._image_size
        self.paths = [
            p.as_posix()
            for ext in self._exts
            for p in Path(f"{self.folder}").glob(f"**/*.{ext}")
        ]

        size = len(self.paths)

        train_size = int(self._split_frac * size)
        test_size = (size - train_size) // 2
        val_size = size - train_size - test_size

        self._train_list, self._test_list, self._val_list = [
            list(val)
            for val in random_split(self.paths, [train_size, test_size, val_size])
        ]

        self._train_dataset = RandomImageSampleDataset(
            image_size=self._image_size,
            path_list=self._train_list,
            num_feature_pixels=self._num_feature_pixels,
            num_target_pixels=self._num_target_pixels,
        )

        self._val_dataset = RandomImageSampleDataset(
            image_size=self._image_size,
            path_list=self._val_list,
            num_feature_pixels=self._num_feature_pixels,
            num_target_pixels=self._num_target_pixels,
        )
        self._test_dataset = RandomImageSampleDataset(
            image_size=self._image_size,
            path_list=self._test_list,
            num_feature_pixels=self._num_feature_pixels,
            num_target_pixels=self._num_target_pixels,
        )

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
