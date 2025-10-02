# Logic Gate Neural Network

Simple feedforward neural network built from scratch in NumPy to classify basic logic gates (AND, OR, NAND, NOR, XOR, XNOR, NOT).

## Features

- Fully implemented from scratch (no Keras/TensorFlow)
- 2-layer neural network with sigmoid activation
- Binary cross-entropy loss
- Gradient descent optimization
- Supports 7 logic gate operations

## Architecture

```
Input Layer (2 neurons)
    ↓
Hidden Layer (2 neurons, sigmoid)
    ↓
Output Layer (1 neuron, sigmoid)
```

## Supported Gates

- AND
- OR
- NAND
- NOR
- XOR
- XNOR
- NOT (single input)

## Usage

```python
import numpy as np

# Set hyperparameters
hiddenneuron = 2
epoch = 100000
learningRate = 0.01

# Input training data
x = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])  # All input combinations
y = np.array([[0, 0, 0, 1]])                 # AND gate outputs

# Initialize and train
parameters = initialparam(inputfeatures, hiddenneuron, outputfeatures)

for i in range(epoch):
    loss, cache, a2 = forwardprop(x, y, parameters)
    gradients = backwardprop(x, y, cache)
    parameters = updateparam(parameters, gradients, learningRate)

# Predict
_, _, predictions = forwardprop(x, y, parameters)
predictions = (predictions > 0.5).astype(int)
```

## Training Example

```
Epoch 0, Loss: 1.188344
Epoch 10000, Loss: 0.094079
Epoch 20000, Loss: 0.034346
Epoch 30000, Loss: 0.019777
Epoch 40000, Loss: 0.013613
Epoch 50000, Loss: 0.010278
...
Final output for AND gate:
[[0 0 0 1]]
```

## Implementation Details

### Forward Propagation
```python
z1 = w1 * x + b1
a1 = sigmoid(z1)
z2 = w2 * a1 + b2
a2 = sigmoid(z2)
loss = -mean(y * log(a2) + (1-y) * log(1-a2))
```

### Backward Propagation
```python
d2 = a2 - y
dw2 = d2 * a1.T / m
db2 = sum(d2)
da1 = w2.T * d2
d1 = da1 * a1 * (1 - a1)
dw1 = d1 * x.T / m
db1 = sum(d1)
```

### Parameter Update
```python
w = w - learning_rate * dw
b = b - learning_rate * db
```

## Truth Tables

| Input A | Input B | AND | OR | NAND | NOR | XOR | XNOR |
|---------|---------|-----|----|----- |-----|-----|------|
| 0       | 0       | 0   | 0  | 1    | 1   | 0   | 1    |
| 0       | 1       | 0   | 1  | 1    | 0   | 1   | 0    |
| 1       | 0       | 0   | 1  | 1    | 0   | 1   | 0    |
| 1       | 1       | 1   | 1  | 0    | 0   | 0   | 1    |

## Requirements

```bash
pip install numpy
```

## Run

Open `ann.ipynb` in Jupyter Notebook or Google Colab and execute cells sequentially.

## Hyperparameters

- Hidden neurons: 2
- Epochs: 100,000
- Learning rate: 0.01
- Activation: Sigmoid
- Loss: Binary cross-entropy

## Notes

- XOR requires at least 2 hidden neurons (not linearly separable)
- Simple gates (AND, OR) converge faster than XOR
- Loss decreases logarithmically with training
