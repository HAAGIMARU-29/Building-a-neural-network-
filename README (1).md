# Deep Neural Network from Scratch — Assignment 3

A NumPy-based implementation of the building blocks of a **deep, fully connected neural network**, including parameter initialization, forward propagation, binary cross-entropy loss, backpropagation, and gradient-descent parameter updates.

This is a learning-focused implementation of the neural-network components from the *Neural Networks and Deep Learning* course (Deep Learning Specialization). It uses helper functions for ReLU and sigmoid activations and their derivatives.

> **Status:** Educational work in progress. The uploaded `assign3.py` defines the individual functions but does **not** include a training loop, trained model, dataset evaluation, or measured accuracy. It also contains a few issues that must be corrected before execution (see [Known issues](#known-issues-in-the-current-file)).

## What it covers

- Parameter initialization for a two-layer or arbitrary-depth neural network
- Layer-wise linear transformations and ReLU/sigmoid activations
- Forward propagation across multiple layers
- Binary cross-entropy cost computation
- Backpropagation and gradients for weights, biases, and activations
- Gradient-descent updates of learned parameters

## Network architecture

The intended L-layer network consists of ReLU hidden layers and a sigmoid output layer for binary classification:

```text
Input X
   │
   ▼
[Linear → ReLU]  × (L − 1)
   │
   ▼
[Linear → Sigmoid]
   │
   ▼
Predicted probability AL
   │
   ▼
Binary cross-entropy loss
   │
   ▼
Backpropagation → Gradients → Parameter updates
```

For layer `l`, the linear computation and activation are:

```text
Z[l] = W[l] @ A[l-1] + b[l]
A[l] = ReLU(Z[l])       # hidden layers
A[L] = sigmoid(Z[L])    # output layer
```

## File structure

```text
.
├── assign3.py         # Neural network building blocks
├── dnn_utils_v2.py    # External activation helpers (downloaded in Colab)
├── testCases_v2.py    # External course-style test cases (downloaded in Colab)
└── README.md
```

Only `assign3.py` was supplied with this project. The two helper modules are fetched by commands at the beginning of that file; they are not included in this submission.

## Functions implemented

| Function | Purpose |
| --- | --- |
| `initialise_parameters(n_x, n_h, n_y)` | Initialize a two-layer network's parameters |
| `initialise_parameters_deep(layer_dims)` | Initialize weights and biases for an L-layer network |
| `linear_forward(A, W, b)` | Calculate a layer's pre-activation output |
| `linear_activation_forward(A_prev, W, b, activation)` | Apply linear transformation followed by ReLU or sigmoid |
| `L_model_forward(X, parameters)` | Propagate inputs through all layers |
| `compute_cost(AL, Y)` | Calculate binary cross-entropy |
| `linear_backward(dZ, cache)` | Compute the linear layer's gradients |
| `linear_activation_backward(dA, cache, activation)` | Backpropagate through the chosen activation and linear layer |
| `L_model_backward(AL, Y, caches)` | Backpropagate through the complete network |
| `update_parameters(params, grads, learning_rate)` | Apply gradient-descent updates |

## Requirements

- Python 3
- NumPy
- Matplotlib
- h5py
- SciPy
- The external `dnn_utils_v2.py` and `testCases_v2.py` modules

Install the Python dependencies using:

```bash
python -m pip install numpy matplotlib h5py scipy
```

## Setup

### Google Colab

The supplied source was exported from a Colab notebook and includes notebook-style shell commands (`!wget`, `!ls`, `!head`). Those commands are intended for notebook cells, not ordinary Python scripts. Open the original notebook, or move these commands into separate Colab cells before running the code.

Original notebook: [Assignment 3 on Google Colab](https://colab.research.google.com/drive/1coz-1WPiT1fzX5Ny6J0FipORe6UTbDEk)

### Local Python

1. Install the dependencies above.
2. Download the helper modules referenced at the top of `assign3.py`:

   ```bash
   curl -L -o dnn_utils_v2.py "https://raw.githubusercontent.com/amanchadha/coursera-deep-learning-specialization/master/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%204/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step/dnn_utils_v2.py"
   curl -L -o testCases_v2.py "https://raw.githubusercontent.com/kathanvyas/Shallow-Neural-Networks/master/testCases_v2.py"
   ```

3. Remove or convert the notebook shell-command lines in `assign3.py`.
4. Correct the issues described below before attempting to run or test the functions.

**Note:** Running the file alone will not train a model or display results because there is no top-level training/evaluation pipeline.

## Known issues in the current file

The implementation is a useful starting point, but these issues are present in the submitted source:

1. `np.zeroes(...)` in `initialise_parameters` should be `np.zeros(...)`.
2. `linear_forward` uses `np.dot(A, W) + b`; with the documented input and weight shapes, the multiplication order should be `np.dot(W, A) + b`.
3. `L_model_forward` reuses `cache` both as the list and as a single layer's cache; it should maintain a separate `caches` list and append each layer cache to it.
4. `linear_backward` uses `np.dot(A_prev.T, dZ)`; the intended weight gradient is `np.dot(dZ, A_prev.T) / m`.
5. `linear_backward` uses `keepDims` (incorrect keyword capitalization) and squeezes `db`, which can lose the required column-vector shape. Use `keepdims=True` and retain shape `(n_l, 1)`.
6. The current binary cross-entropy cost can encounter `log(0)` when predicted probabilities are exactly 0 or 1; numerical clipping can improve stability.

These issues are documented rather than silently changing the submitted code.

## Next steps

- Fix the listed bugs and validate each function with the imported test cases.
- Add an end-to-end training loop and a prediction function.
- Train on a documented binary-classification dataset.
- Plot training loss and report evaluation metrics.
- Add automated tests and a reproducible example.

## Acknowledgments

- [Deep Learning Specialization — DeepLearning.AI](https://www.deeplearning.ai/courses/deep-learning-specialization/)
- [External activation helper source](https://github.com/amanchadha/coursera-deep-learning-specialization)
- [External test-case source](https://github.com/kathanvyas/Shallow-Neural-Networks)
