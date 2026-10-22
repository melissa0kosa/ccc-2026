# Neural Networks from Scratch (NumPy)

Expect levels like these: *"here are the weights, compute the outputs"*, *"implement the forward pass"*, *"find weights so the network outputs X"*, or *"train a perceptron"*. `ccc_2026/numpy_nn.py` covers them, and it is tested against PyTorch.

## Forward pass
```python
from ccc_2026.numpy_nn import forward
out = forward(X, weights=[W1, W2], biases=[b1, b2], activations="relu", output="softmax")
```
A dense layer computes `z = a @ W + b` with W shaped `(n_in, n_out)`.

**Read the statement's convention.** Many statements (and PyTorch) store `W` as `(n_out, n_in)` and compute `W @ x + b`. In that case pass `W.T`. A quick test: the shapes must chain, and the sample input must reproduce the sample output.

## Activations
| Name | Formula | Derivative |
|---|---|---|
| ReLU | max(0, z) | 1 if z > 0 else 0 |
| Leaky ReLU | z if z > 0 else αz | 1 or α |
| Sigmoid | 1/(1+e^-z) | σ(1-σ) |
| Tanh | tanh z | 1 - tanh² |
| Softmax | e^z_i / Σe^z_j (subtract the max first!) | used with CE: dL/dz = p - y |
| Step / Heaviside | 1 if z ≥ 0 else 0 | classic perceptron (check the ≥ vs > rule in the statement) |

## Backpropagation (2-layer, softmax + cross-entropy)
```python
Z1 = X @ W1 + b1; H = np.maximum(0, Z1)
Z2 = H @ W2 + b2; P = softmax(Z2)
dZ2 = (P - Y_onehot) / N
dW2 = H.T @ dZ2;  db2 = dZ2.sum(0)
dH  = dZ2 @ W2.T; dZ1 = dH * (Z1 > 0)
dW1 = X.T @ dZ1;  db1 = dZ1.sum(0)
W -= lr * dW  # or Adam
```
`NumpyMLP(sizes=[n_in, 64, n_out], output="softmax").fit(X, y)` implements this with Adam.

**Gradient check** when you hand-write gradients: compare with the numerical value `(L(w+ε) - L(w-ε)) / 2ε` at ε = 1e-5. The relative error should be < 1e-6.

## Classic perceptron (if asked)
```python
w = np.zeros(d); b = 0
for x, t in zip(X, T):                 # t in {-1, +1}
    if t * (x @ w + b) <= 0:
        w += lr * t * x; b += lr * t
```
It only converges if the data is linearly separable.

## Designing weights by hand
- One neuron with a step or sigmoid activation is a **half-plane** `w·x + b > 0`.
  - AND(x1,x2): w = (1,1), b = -1.5
  - OR: w = (1,1), b = -0.5
  - NOT: w = -1, b = 0.5
- **XOR** needs a hidden layer: `h1 = relu(x1 + x2)`, `h2 = relu(x1 + x2 - 1)`, `y = h1 - 2*h2`.
- A ReLU network is **piecewise linear**. `relu(x - a)` adds a kink at `a`, so any 1-D piecewise linear function is `c0 + c1*x + Σ k_i * relu(x - a_i)`.
- Max of two values: `max(a,b) = b + relu(a - b)`. Absolute value: `|x| = relu(x) + relu(-x)`.
- Identity through ReLU: `x = relu(x) - relu(-x)`.
- With sigmoid, scale up the weights (×100) to approximate a step function.

## Numerical precision
- Use `float64` in NumPy unless the statement says float32. If the expected output was made with float32, use `X.astype(np.float32)` throughout to match the rounding.
- Follow the stated rounding exactly: `f"{x:.4f}"` vs `round()` (banker's rounding on .5!) vs truncation.
- Softmax: subtract the row max to avoid overflow.

## Loading weights from files
- Weights may be given as text matrices. Parse each layer, then **check the shapes chain** (`W1.shape[1] == W2.shape[0]`).
- If the weights are a PyTorch `state_dict`, use `from_torch_state_dict(sd)` (transposes for you).
