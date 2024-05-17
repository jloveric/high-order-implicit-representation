import pytest
from high_order_implicit_representation.single_image_dataset import (
    image_to_dataset,
    ImageNeighborhoodReader,
    image_neighborhood_dataset,
    Text2ImageDataset,
    Text2ImageRenderDataset,
)
import torch
from torch.utils.data import Dataset, DataLoader


def test_neighborhood_dataset():
    factor = 11
    size = factor * 9
    image = torch.arange(3 * size * size)
    image = image.reshape(3, size, size)
    features, targets, _, _, _ = image_neighborhood_dataset(
        image=image, width=3, outside=3, stride=3
    )

    expected = (size - 2 * 3) // 3

    assert features.shape[0] == expected * expected
    assert features.shape[1] == 9 * 9 * 3 - 3 * 3 * 3
    assert targets.shape[0] == expected * expected
    assert targets.shape[1] == 27


def test_image_neighborhood_reader():
    ind = ImageNeighborhoodReader(filename="images/0000.jpg", width=3, outside=1)

    assert ind.targets.shape == torch.Size([784, 27])
    assert ind.features.shape == torch.Size([784, 48])
    assert ind.lastx == 27
    assert ind.lasty == 27
    assert ind.image.shape == torch.Size([3, 32, 32])


def test_parquet_dataset():
    dataset = Text2ImageDataset(filenames=["test_data/test.parquet"])
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # caption, position, rgb = next(iter(dataloader))
    for batch in dataloader:
        print("batch", batch)
        break
    # print('val', caption, position, rgb)


def test_text_to_image_sampler_dataloader():
    dataset = Text2ImageRenderDataset(filenames=["test_data/test.parquet"])

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    for element in dataloader:
        print("element", element[0][0].shape)
        print("element", element[0][1].shape)
        print("element", element[1][0].shape)
        assert element[0].shape[0] ==2
        assert len(element) == 3
        break
