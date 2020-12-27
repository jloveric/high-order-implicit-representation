# implicit-representation
Implicit representation of various things using PyTorch and high order layers

Train a model
```
python implicit_images.py mlp.hidden.width=10 mlp.hidden.layers=2 lr=1e-3 mlp.n=3 mlp.periodicity=2.0 mlp.layer_type=continuous mlp.hidden.segments=2 mlp.input.segments=100 mlp.output.segments=2
```

Evaluate a model example
```
python implicit_images.py train=False checkpoint=outputs/2020-12-26/15-42-25/lightning_logs/version_0/checkpoints/'epoch=8-step=190731.ckpt'
```