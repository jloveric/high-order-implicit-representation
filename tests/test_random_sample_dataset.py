import pytest
from high_order_implicit_representation.random_sample_dataset import (
    RandomImageSampleDataset,
    random_image_sample_collate_fn,
    RandomImageSampleDataModule,
    random_symmetric_sample,
    RadialRandomImageSampleDataset
)
import torch

def test_radial_random_image_sample_dataset_specific():
    num_feature_pixels = 25
    num_target_pixels = 1

    dataset = RadialRandomImageSampleDataset(
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

    features, targets = next(this_iter)
    print('features', features)
    print('targets', targets)
    assert features.shape[0] == targets.shape[0] == 2048
    assert features.shape[1] == num_feature_pixels
    assert targets.shape[1] == num_target_pixels
    assert features.shape[2] == 5
    assert targets.shape[2] == 3


def test_random_symmetric_sample():

    samples = torch.tensor([[5, 5], [0, 0], [10, 0], [0, 10], [10, 10]])
    image_size = 11
    interp_size = 5
    result = random_symmetric_sample(
        image_size=image_size, interp_size=interp_size, samples=samples
    )
    print('result', result)
    assert torch.all(result < image_size).tolist() is True
    assert torch.all(result >= 0).tolist() is True


def test_random_image_sample_dataset_specific():
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

    features, targets = next(this_iter)

    print("features", features)
    print("targets", targets)

    assert features.shape[0] == targets.shape[0] == 78
    assert features.shape[1] == num_feature_pixels
    assert targets.shape[1] == num_target_pixels
    assert features.shape[2] == 5
    assert targets.shape[2] == 3


@pytest.mark.parametrize("num_feature_pixels", [1, 3, 75])
@pytest.mark.parametrize("num_target_pixels", [1])
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

    features, targets = next(this_iter)

    assert features.shape[0] == targets.shape[0]
    assert features.shape[1] == num_feature_pixels
    assert targets.shape[1] == num_target_pixels
    assert features.shape[2] == 5
    assert targets.shape[2] == 3


def test_random_image_sample_datamodule():

    dataset = RandomImageSampleDataModule(
        image_size=32,
        folder="images",
        num_feature_pixels=25,
        num_target_pixels=1,
    )

    dataset.setup()

    assert len(dataset.train_dataset) > 0

    dataloader = dataset.train_dataloader()

    features, targets = next(iter(dataloader))

    assert features.shape[1] == 25
    assert targets.shape[1] == 1
    assert features.shape[2] == 5
    assert targets.shape[2] == 3


