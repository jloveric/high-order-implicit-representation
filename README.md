# Implicit Representation with High Order Layers
Implicit representation of various things using PyTorch and high order layers.  The network uses high order layers as implemented [here](https://github.com/jloveric/high-order-layers-torch).  Implicit representation is a fancy way of saying creating a function that fits the training set (ignoring generalization), however you may end up with a compact representation of the original data and a function that interpolates the data.  Neural networks, and especially high-order networks are good at this problem.

# Implicit Representation of Images

Train a model
```
python implicit_images.py mlp.hidden.width=10 mlp.hidden.layers=2 lr=1e-3 mlp.n=3 mlp.periodicity=2.0 mlp.layer_type=continuous mlp.hidden.segments=2 mlp.input.segments=100 mlp.output.segments=2 batch_size=256

```

Evaluate a model example
```
python implicit_images.py train=False checkpoint=\"multirun/2021-01-10/18-31-32/0/lightning_logs/version_0/checkpoints/epoch=49-step=145349.ckpt\" rotations=2
```
## Examples
### Piecewise Continuous
The example below uses piecewise quadratic polynomials.  The input layer is the x, y position where there are 100 segments
for each input connection.  There is 1 hidden layers with 40 units each and 2 segments.  There are 3 outputs representing the RGB colors, where each output has 2 segment.  In total there are 40.8k parameters,
The raw image can be represented by 2.232e6 8bit parameters.
```python
python implicit_images.py -m mlp.hidden.width=40 mlp.hidden.layers=1 lr=1e-3 mlp.n=3 mlp.periodicity=2.0 mlp.layer_type=continuous mlp.hidden.segments=2 mlp.input.segments=100 mlp.output.segments=2 batch_size=256 mlp.input.width=4 rotations=2
```
![Piecewise continuous polynomial network.](results/100x40x1hidden.png)
### Fourier Series
similarly with a fourier series network
```python
python implicit_images.py -m mlp.hidden.width=40 mlp.hidden.layers=1 lr=1e-3 mlp.n=3 mlp.n_in=31 mlp.layer_type=fourier batch_size=256 mlp.input.width=4 rotations=2
```
![Fourier series network.](results/100x40x1hidden.fourier.png)
### Piecewise Discontinuous
and discontinuous polynomial
```python
python implicit_images.py -m mlp.hidden.width=40 mlp.hidden.layers=1 lr=1e-3 mlp.n=3 mlp.periodicity=2.0 mlp.layer_type=discontinuous mlp.hidden.segments=2 mlp.input.segments=100 mlp.output.segments=2 batch_size=256 mlp.input.width=4 rotations=2
```
![Piecewise discontinuous network.](results/100x40x1hidden.discontinuous.png)

# Implicit Neighborhoods
Train interpolator / extrapolator
```
python implicit_neighborhood.py mlp.hidden.width=10 mlp.hidden.layers=2 lr=1e-3 mlp.n=3 mlp.periodicity=2.0 mlp.layer_type=continuous mlp.hidden.segments=2 mlp.input.segments=100 mlp.output.segments=2 batch_size=256
```
create output with trained filter
```
python implicit_neighborhood.py train=False checkpoint=<>
```
Training on the image of the newt and applying to the image of jupiter gives
the following results.
![Piecewise Polynomial Newt to Jupiter.](results/salamander_to_jupiter.png)

# Implicit Representation of Books (Text), (Language Interpolation)
Run with this command
```
python language_interpolation.py 
```
running with Nevergrad
```
python language_interpolation.py hydra/sweeper=nevergrad --cfg hydra -p hydra.sweeper
```
## Apply a model
```
python language_interpolation.py train=False checkpoint=\"multirun/2021-05-16/17-27-58/2/lightning_logs/version_0/checkpoints/epoch=19-step=34199.ckpt\" topk=2 num_predict=200 text="The stars were"
```
example output (model trained to predict the next character given the preceeding 10) using a single hidden layer
```
prompt: The stars were
output: The stars were dreams of my friends and the secret of my father the expenses, in the morning when I awill a trees than sunsitice of my friends when I continued, and the deatures, and the destruction of distance of 
```
The model attempts to memorize the entire book (Frankenstein) by predicting the next character. Each character is provided as a probability by the network.  By choosing (weighted by probability) between the top 2 next characters you produce text (so far nonsense) that changes every time the function is called.  This is fairly standard, but we deliberately memorize the training set, there is no test set.
# Run tests
```
pytest test.py -s
```