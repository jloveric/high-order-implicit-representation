import pytest
from high_order_implicit_representation.rendering import neighborhood_sample_generator
import torch
from omegaconf import DictConfig
from high_order_implicit_representation.networks import Net

"""
@pytest.mark.parametrize("image_height", [32])
@pytest.mark.parametrize("image_width", [32])
@pytest.mark.parametrize("width", [5])
@pytest.mark.parametrize("outside", [5])
"""


def test_neighborhood_sample_generator():  # (image_height, image_width, width, outside):
    image_height = 32
    image_width = 32
    width = 3
    outside = 3

    input_features = ((width + 2 * outside) * (width + 2 * outside) - width * width) * 3
    output_size = width * width * 3

    cfg = DictConfig(
        content={
            "max_epochs": 1,
            "gpus": 0,
            "lr": 1e-4,
            "batch_size": 16,
            "segments": 2,
            "optimizer": {
                "name": "adam",
                "lr": 1.0e-3,
                "scheduler": "plateau",
                "patience": 10,
                "factor": 0.1,
            },
            "mlp": {
                "model_type": "high_order",
                "layer_type": "discontinuous",
                "normalize": True,
                "features": input_features,
                "n": 2,
                "n_in": 2,
                "n_out": None,
                "n_hidden": None,
                "periodicity": 2.0,
                "rescale_output": False,
                "input": {
                    "segments": 100,
                    "width": input_features,
                },
                "output": {
                    "segments": 2,
                    "width": output_size,
                },
                "hidden": {
                    "segments": 2,
                    "layers": 2,
                    "width": 10,
                },
            },
        }
    )

    model = Net(cfg)
    image = torch.rand(3, image_height, image_width) * 2 - 1

    result = neighborhood_sample_generator(
        model=model,
        image=image,
        width=width,
        outside=outside,
        output_size=[image_height, image_width],
    )

    print("result.shape", result.shape)
    assert result.shape == image.shape