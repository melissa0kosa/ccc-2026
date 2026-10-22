# Neural Networks Playbook

Cloudflight describes the 2026 Data & AI track with *"experimenting with neural networks"*. This page is the main reference.

## When an NN is the right tool
- The level says to build or train a neural network, or gives you one.
- Data is **raw signals**: images and pixels, audio, text or character sequences, long time series.
- Boosting plateaus below the threshold on a smooth, high-dimensional function.
- For plain tabular data, try LightGBM first. An NN mainly adds value in an ensemble.

## Default recipe (works for 90% of cases)
```python
from ccc_2026.nn import MLP, fit, predict, seed_everything
from sklearn.preprocessing import StandardScaler

seed_everything(0)
sx = StandardScaler().fit(X_tr)                                       # fit on train only
Xtr, Xva = sx.transform(X_tr), sx.transform(X_va)
ym, ys = y_tr.mean(), y_tr.std()                                      # scale regression targets too
model, hist = fit(MLP(Xtr.shape[1], 1, hidden=(512, 256, 128), dropout=0.1),
                  Xtr, (y_tr - ym) / ys, Xva, (y_va - ym) / ys,
                  task="regression", epochs=200, lr=3e-3, batch_size=512, patience=20)
pred = predict(model, Xva).ravel() * ys + ym
```
`fit` uses **AdamW + OneCycleLR (stepped per batch) + bf16 autocast + gradient clipping + early stopping that keeps the best weights**. Data lives fully on the GPU and batches are sliced directly, with no DataLoader overhead.

## Hyperparameter cheat-sheet
| Knob | Start | Range | Notes |
|---|---|---|---|
| Learning rate | 3e-3 (OneCycle max) | 1e-4 – 1e-2 | **Most important knob.** Too high: loss NaN or jumping. Too low: slow, flat |
| Batch size | 512 | 64 – 4096 | Larger needs a higher LR; small data needs a smaller batch |
| Width × depth | 3 × 256–512 | 2–4 layers | Wider beats deeper for tabular data |
| Dropout | 0.1 | 0 – 0.5 | Raise it if train ≪ val loss |
| Weight decay | 1e-4 | 1e-5 – 1e-2 | AdamW decoupled decay |
| Epochs | 200 + early stop | — | patience 10–30 |
| Activation | GELU / ReLU | SiLU | Rarely matters |
| Norm | BatchNorm | LayerNorm for tiny batches or sequences | |

## Inputs and targets
- **Always standardize inputs** (`StandardScaler`). Unscaled features are the #1 reason an NN "doesn't learn".
- Use `QuantileTransformer(output_distribution="normal")` for heavy-tailed features.
- Categoricals: one-hot if there are few values, otherwise `nn.Embedding(n_cat, min(50, n_cat//2))`.
- Regression targets: standardize them, or `log1p` if skewed. **Invert before writing outputs.**
- Classification: integer labels 0..K-1 with `CrossEntropyLoss` on raw logits. Do **not** add a softmax before the loss.
- For imbalanced classes or macro-F1: `CrossEntropyLoss(weight=class_weights)` or oversampling.
- Multi-label: `BCEWithLogitsLoss`.

## Loss functions
| Task | Loss | Output layer |
|---|---|---|
| Regression, RMSE metric | `MSELoss` | Linear |
| Regression, MAE metric or outliers | `L1Loss` / `HuberLoss` | Linear |
| Multiclass | `CrossEntropyLoss` (label_smoothing=0.05 can help) | Linear (logits) |
| Binary / multi-label | `BCEWithLogitsLoss` | Linear (logits) |

## Debugging a network that doesn't learn
1. **Overfit one batch:** train on 32 samples. Loss should approach 0. If it doesn't, there's a bug (shapes, labels, LR).
2. **Check shapes:** `(N,)` vs `(N,1)` targets silently broadcast in MSE and give garbage. `fit` reshapes them for you.
3. **Loss is NaN:** lower the LR, check for NaN or inf in the inputs (`np.isfinite(X).all()`), keep gradient clipping on.
4. **Val loss rises early:** overfitting. Add dropout and weight decay, use a smaller net, augment the data.
5. **Train and val both stuck high:** underfitting. Use a bigger net, higher LR, more epochs, better features, scaled inputs.
6. **Large gap between a good CV and a failed submission:** check the inverse target transform, `model.eval()` (dropout off), and the output row order.
7. **LR range test:** train a few epochs at 1e-4, 1e-3, 1e-2 and pick the best that is stable.

## Speed on the RTX 5070 Ti (Blackwell, sm_120)
- The env uses torch 2.14 + CUDA 13 (`cu130` wheels), which include sm_120. `uv run ccc gpu` checks this.
- bf16 autocast is on by default in `fit`. TF32 matmuls are enabled in `get_device()`.
- Tabular data under 1 M rows: keep everything on the GPU and slice batches (as `fit` does).
- Images and audio: use a `DataLoader` with `num_workers=0` on Windows (multiprocessing is slow or fragile there), or pre-compute features as tensors.
- `torch.compile(model)` only helps on long runs. Skip it in a contest (compile time, plus Windows needs Triton).

## Ensembling NNs
- Average 3–5 seeds, or the 5 fold models, for an easy +1–3%.
- Blend with LightGBM: `0.5*nn + 0.5*gbm`, with weights tuned on the out-of-fold predictions.

## Final model
- Option A: average the K fold models' predictions on test. Safe, and needs no retraining.
- Option B: retrain on all data for about `best_epoch × 1.1` epochs with no validation set (`fit(..., X_va=None)` runs all epochs).

See [NN Architectures by Data Type](NN-Architectures-by-Data-Type) and [Neural Networks from Scratch](Neural-Networks-from-Scratch).
