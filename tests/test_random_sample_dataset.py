import pytest
from high_order_implicit_representation.random_sample_dataset import (
    RandomImageSampleDataset,
)
import torch


def test_random_image_sample_dataset():
    RandomImageSampleDataset(
        image_size=32,
        path_list=["images/0000.jpg"],
        num_train_pixels=25,
        num_target_pixels=1,
    )
