import pytest
from high_order_implicit_representation.single_image_dataset import (
    image_to_dataset,
    ImageNeighborhoodReader,
)
import torch


def test_image_neighborhood_reader():
    ind = ImageNeighborhoodReader(filename="images/0000.jpg")

    assert ind.features.shape == torch.Size([729, 27])
    assert ind.targets.shape == torch.Size([729, 48])
    assert ind.lastx == 27
    assert ind.lasty == 27
    assert ind.image.shape == torch.Size([32, 32, 3])
