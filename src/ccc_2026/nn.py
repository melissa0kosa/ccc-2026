"""PyTorch helpers: device setup, a tabular MLP and a compact training loop with early stopping.

Typical use (tabular):
    model = MLP(n_in=X.shape[1], n_out=1)
    model, hist = fit(model, X_tr, y_tr, X_va, y_va, task="regression")
    pred = predict(model, X_test)
"""

from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


class MLP(nn.Module):
    """Tabular MLP: [Linear -> norm -> GELU -> Dropout] x len(hidden) -> Linear."""

    def __init__(self, n_in: int, n_out: int, hidden=(512, 256, 128), dropout: float = 0.1,
                 norm: str = "batch"):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if norm == "batch":
                layers.append(nn.BatchNorm1d(h))
            elif norm == "layer":
                layers.append(nn.LayerNorm(h))
            layers += [nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _to_tensor(a, device, dtype=torch.float32):
    if isinstance(a, torch.Tensor):
        return a.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)


def fit(model: nn.Module, X_tr, y_tr, X_va=None, y_va=None, *, task: str = "regression",
        epochs: int = 200, batch_size: int = 512, lr: float = 3e-3, weight_decay: float = 1e-4,
        patience: int = 20, loss_fn: nn.Module | None = None, device=None, amp: bool = True,
        verbose: int = 10):
    """Train with AdamW + OneCycleLR, bf16 autocast on GPU, early stopping on validation loss.

    task: "regression" (y float, shape (N,) or (N,k)) or "classification" (y int class ids).
    Data is kept fully on the device and batches are sliced directly (fast for tabular data).
    Returns (best_model, history).
    """
    device = device or get_device()
    model = model.to(device)
    cls = task == "classification"
    y_dtype = torch.long if cls else torch.float32
    Xt, yt = _to_tensor(X_tr, device), _to_tensor(y_tr, device, y_dtype)
    if not cls and yt.ndim == 1:
        yt = yt[:, None]
    has_val = X_va is not None
    if has_val:
        Xv, yv = _to_tensor(X_va, device), _to_tensor(y_va, device, y_dtype)
        if not cls and yv.ndim == 1:
            yv = yv[:, None]
    loss_fn = loss_fn or (nn.CrossEntropyLoss() if cls else nn.MSELoss())

    n = len(Xt)
    steps_per_epoch = max(1, (n + batch_size - 1) // batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                steps_per_epoch=steps_per_epoch)
    use_amp = amp and device.type == "cuda"

    best_loss, best_state, bad = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            if len(idx) < 2 and any(isinstance(m, nn.BatchNorm1d) for m in model.modules()):
                continue  # BatchNorm cannot train on a batch of one
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(Xt[idx])
            loss = loss_fn(out.float(), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        history["train_loss"].append(total / n)

        monitor = history["train_loss"][-1]
        if has_val:
            monitor = evaluate_loss(model, Xv, yv, loss_fn, use_amp)
            history["val_loss"].append(monitor)
        if monitor < best_loss - 1e-7:
            best_loss, best_state, bad = monitor, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose and (epoch % verbose == 0 or epoch == epochs - 1):
            val = f" val={history['val_loss'][-1]:.5f}" if has_val else ""
            print(f"epoch {epoch:4d} train={history['train_loss'][-1]:.5f}{val} "
                  f"({time.time() - t0:.0f}s)")
        if has_val and bad >= patience:
            if verbose:
                print(f"early stop at epoch {epoch}, best val={best_loss:.5f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate_loss(model, X, y, loss_fn, use_amp: bool = False, batch_size: int = 8192) -> float:
    model.eval()
    total = 0.0
    for i in range(0, len(X), batch_size):
        with torch.autocast(X.device.type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(X[i:i + batch_size])
        total += loss_fn(out.float(), y[i:i + batch_size]).item() * len(out)
    return total / len(X)


@torch.no_grad()
def predict(model: nn.Module, X, batch_size: int = 8192, proba: bool = False) -> np.ndarray:
    """Raw outputs (regression), or class ids / softmax probabilities (classification)."""
    device = next(model.parameters()).device
    model.eval()
    Xt = _to_tensor(X, device)
    outs = [model(Xt[i:i + batch_size]).float() for i in range(0, len(Xt), batch_size)]
    out = torch.cat(outs)
    if proba:
        return torch.softmax(out, dim=1).cpu().numpy()
    return out.cpu().numpy()


def predict_classes(model: nn.Module, X, batch_size: int = 8192) -> np.ndarray:
    return predict(model, X, batch_size).argmax(axis=1)
