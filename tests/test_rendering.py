import pytest
from high_order_implicit_representation.rendering import (
    neighborhood_sample_generator,
    NeighborGenerator,
)
import torch
from omegaconf import DictConfig
from high_order_implicit_representation.networks import Net
from pytorch_lightning import Trainer


@pytest.mark.parametrize("image_height", [21, 32])
@pytest.mark.parametrize("image_width", [21, 32])
@pytest.mark.parametrize("width", [1, 3, 5])
@pytest.mark.parametrize("outside", [1, 3, 5])
def test_neighborhood_sample_generator(image_height, image_width, width, outside):

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
    )

    assert result.shape == image.shape


def test_neighbor_generator():
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
    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        gpus=cfg.gpus,
    )

    # Just make sure this runs
    generator = NeighborGenerator(
        samples=2, frames=2, output_size=[64, 64], width=3, outside=3
    )
    generator.on_train_epoch_end(trainer=trainer, pl_module=model)
