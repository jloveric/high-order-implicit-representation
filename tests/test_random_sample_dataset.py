import pytest
from high_order_implicit_representation.random_sample_dataset import (
    RandomImageSampleDataset,
    random_image_sample_collate_fn,
)
import torch


def test_random_image_sample_dataset():
    num_feature_pixels = 25
    num_target_pixels = 1

    dataset = RandomImageSampleDataset(
        image_size=32,
        path_list=["images/0000.jpg", "images/jupiter.jpg"],
        num_feature_pixels=num_feature_pixels,
        num_target_pixels=num_target_pixels,
    )

    this_iter = iter(
        torch.utils.data.DataLoader(
            dataset, batch_size=3, collate_fn=random_image_sample_collate_fn
        )
    )

    features, targets = this_iter.next()

    assert features.shape[0] == targets.shape[0] == 78
    assert features.shape[1] == num_feature_pixels
    assert targets.shape[1] == num_target_pixels
    assert features.shape[2] == targets.shape[2]


@pytest.mark.parametrize("num_feature_pixels", [1, 3, 75])
@pytest.mark.parametrize("num_target_pixels", [1, 3, 10])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_random_image_sample_dataset(num_feature_pixels, num_target_pixels, batch_size):

    dataset = RandomImageSampleDataset(
        image_size=32,
        path_list=["images/0000.jpg", "images/jupiter.jpg"],
        num_feature_pixels=num_feature_pixels,
        num_target_pixels=num_target_pixels,
    )

    this_iter = iter(
        torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=random_image_sample_collate_fn
        )
    )

    features, targets = this_iter.next()

    assert features.shape[0] == targets.shape[0]
    assert features.shape[1] == num_feature_pixels
    assert targets.shape[1] == num_target_pixels
    assert features.shape[2] == targets.shape[2]
