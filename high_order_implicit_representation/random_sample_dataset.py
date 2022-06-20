from typing import Optional
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, random_split

import pytorch_lightning as pl
from torchvision import transforms
from PIL import Image
from typing import List
import logging

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    def __init__(
        self,
        image_size: int,
        path_list: List[str],
        num_train_pixels: int = 25,
        num_target_pixels: int = 1,
    ):
        super().__init__()
        self._image_size = image_size
        self._paths = path_list
        self.num_train_pixels = num_train_pixels
        self.num_target_pixels = num_target_pixels

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)

        img = self.transform(img)


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_size: int,
        folder: str,
        exts: List[str] = ["jpg", "jpeg", "png", "JPEG"],
        batch_size=32,
        num_workers=10,
        split_frac=0.8,
    ):
        super().__init__()
        self._image_size = image_size
        self._folder = folder
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

        self._train_dataset = ImageDataset(
            image_size=self._image_size, path_list=self._train_list
        )

        self._val_dataset = ImageDataset(
            image_size=self._image_size, path_list=self._val_list
        )
        self._test_dataset = ImageDataset(
            image_size=self._image_size, path_list=self._test_list
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
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self._num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self._num_workers,
        )
