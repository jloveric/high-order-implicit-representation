import pytest
from high_order_implicit_representation.utils import generate_sample
import torch
from omegaconf import DictConfig
from high_order_implicit_representation.networks import Net


def test_generate_sample():
    sample_points = 10
    input_features = 70

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
                    "width": 3,
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

    results = generate_sample(
        model=model,
        features=sample_points,
        image_size=64,
        targets=1,
        iterations=3,
        rotations=2,
        batch_size=256,
    )
    assert len(results) == 3
    ans = [result.shape == torch.Size([3, 64, 64]) for result in results]
    assert all(ans) is True
