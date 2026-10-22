"""Neural networks in plain NumPy: for levels that hand you weights, forbid libraries,
or ask you to implement forward/backward passes yourself.

Weight convention here: W has shape (n_in, n_out) and a layer computes x @ W + b.
PyTorch nn.Linear stores weight as (n_out, n_in) and computes x @ W.T + b -> transpose it!
"""

from __future__ import annotations

import numpy as np

ACTIVATIONS = {
    "relu": lambda z: np.maximum(0.0, z),
    "leaky_relu": lambda z: np.where(z > 0, z, 0.01 * z),
    "sigmoid": lambda z: 1.0 / (1.0 + np.exp(-z)),
    "tanh": np.tanh,
    "linear": lambda z: z,
    "identity": lambda z: z,
}

DERIVATIVES = {  # in terms of the pre-activation z and the activation a
    "relu": lambda z, a: (z > 0).astype(z.dtype),
    "leaky_relu": lambda z, a: np.where(z > 0, 1.0, 0.01),
    "sigmoid": lambda z, a: a * (1.0 - a),
    "tanh": lambda z, a: 1.0 - a**2,
    "linear": lambda z, a: np.ones_like(z),
    "identity": lambda z, a: np.ones_like(z),
}


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def forward(x: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray],
            activations: list[str] | str = "relu", output: str = "linear") -> np.ndarray:
    """Forward pass of a dense network. `activations` applies to hidden layers.

    output: "linear", "sigmoid", "softmax", or any key of ACTIVATIONS.
    """
    n_hidden = len(weights) - 1
    if isinstance(activations, str):
        activations = [activations] * n_hidden
    a = np.asarray(x, dtype=np.float64)
    for i, (W, b) in enumerate(zip(weights, biases, strict=True)):
        z = a @ W + b
        if i < n_hidden:
            a = ACTIVATIONS[activations[i]](z)
        else:
            a = softmax(z) if output == "softmax" else ACTIVATIONS[output](z)
    return a


def from_torch_state_dict(state_dict) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract (weights, biases) in this module's convention from an nn.Sequential of Linears."""
    ws, bs = [], []
    for k, v in state_dict.items():
        if k.endswith(".weight") and v.ndim == 2:
            ws.append(v.detach().cpu().numpy().T)
        elif k.endswith(".bias") and len(bs) < len(ws):
            bs.append(v.detach().cpu().numpy())
    return ws, bs


class NumpyMLP:
    """Minimal trainable MLP (mini-batch SGD with momentum / Adam), no frameworks."""

    def __init__(self, sizes: list[int], hidden_act: str = "relu", output: str = "linear",
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.hidden_act, self.output = hidden_act, output
        self.W = [rng.normal(0, np.sqrt(2.0 / m), size=(m, n))
                  for m, n in zip(sizes, sizes[1:], strict=False)]
        self.b = [np.zeros(n) for n in sizes[1:]]

    def forward(self, x):
        return forward(x, self.W, self.b, self.hidden_act, self.output)

    def _forward_cache(self, x):
        zs, acts = [], [x]
        a = x
        for i, (W, b) in enumerate(zip(self.W, self.b, strict=True)):
            z = a @ W + b
            last = i == len(self.W) - 1
            if not last:
                a = ACTIVATIONS[self.hidden_act](z)
            elif self.output == "softmax":
                a = softmax(z)
            else:
                a = ACTIVATIONS[self.output](z)
            zs.append(z)
            acts.append(a)
        return zs, acts

    def fit(self, X, y, epochs: int = 200, lr: float = 1e-3, batch_size: int = 64,
            seed: int = 0, verbose: int = 0):
        """Loss: MSE for linear output, cross-entropy for softmax (y = int labels)
        and binary cross-entropy for sigmoid (y in {0,1}). Uses Adam."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if self.output == "softmax":
            Y = np.eye(self.W[-1].shape[1])[y.astype(int)]
        else:
            Y = y.reshape(len(y), -1).astype(np.float64)
        rng = np.random.default_rng(seed)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0
        for epoch in range(epochs):
            order = rng.permutation(len(X))
            for s in range(0, len(X), batch_size):
                idx = order[s:s + batch_size]
                zs, acts = self._forward_cache(X[idx])
                n = len(idx)
                # softmax+CE, sigmoid+BCE and linear+MSE all give (a - y) as dL/dz (up to 2x)
                delta = (acts[-1] - Y[idx]) / n
                gW, gb = [None] * len(self.W), [None] * len(self.b)
                for layer in range(len(self.W) - 1, -1, -1):
                    gW[layer] = acts[layer].T @ delta
                    gb[layer] = delta.sum(axis=0)
                    if layer > 0:
                        delta = (delta @ self.W[layer].T) * DERIVATIVES[self.hidden_act](
                            zs[layer - 1], acts[layer])
                t += 1
                for i, g in enumerate(gW + gb):
                    m[i] = 0.9 * m[i] + 0.1 * g
                    v[i] = 0.999 * v[i] + 0.001 * g * g
                    mhat, vhat = m[i] / (1 - 0.9**t), v[i] / (1 - 0.999**t)
                    params[i] -= lr * mhat / (np.sqrt(vhat) + 1e-8)
            if verbose and epoch % verbose == 0:
                print(f"epoch {epoch}: loss={self.loss(X, y):.5f}")
        return self

    def loss(self, X, y) -> float:
        p = self.forward(X)
        if self.output == "softmax":
            return float(-np.mean(np.log(p[np.arange(len(y)), np.asarray(y, int)] + 1e-12)))
        Y = np.asarray(y, dtype=np.float64).reshape(len(y), -1)
        if self.output == "sigmoid":
            return float(-np.mean(Y * np.log(p + 1e-12) + (1 - Y) * np.log(1 - p + 1e-12)))
        return float(np.mean((p - Y) ** 2))

    def predict(self, X):
        p = self.forward(X)
        if self.output == "softmax":
            return p.argmax(axis=1)
        return p.ravel() if p.shape[1] == 1 else p
