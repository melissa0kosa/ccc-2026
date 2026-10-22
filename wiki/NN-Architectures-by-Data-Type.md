# NN Architectures by Data Type

Copy-paste PyTorch modules. All of them work with `ccc_2026.nn.fit` (any `nn.Module` taking a float tensor). For integer token inputs, write a short custom loop or cast inside `forward`.

## Tabular: MLP
`ccc_2026.nn.MLP(n_in, n_out, hidden=(512,256,128), dropout=0.1)`. See [Neural Networks Playbook](Neural-Networks-Playbook).

## Sequences: strings, paths, event logs
Example: in 2022, a 30-character string over the letters a–i mapped to 5 classes, solved with LSTM 256 → LSTM 128 → Dense at ~0.99 macro-F1.

```python
class SeqClassifier(nn.Module):
    def __init__(self, vocab, n_classes, emb=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.rnn = nn.GRU(emb, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Linear(2 * hid, n_classes)
    def forward(self, x):                      # x: (B, T) float -> cast to long
        h, _ = self.rnn(self.emb(x.long()))
        return self.head(h.mean(dim=1))        # mean-pool over time
```
- Tokenize: `vocab = {c: i+1 for i, c in enumerate(sorted(set("".join(seqs))))}`. 0 is padding.
- A **1D-CNN** is often as good and faster: `Conv1d(emb, 128, 5, padding=2) → ReLU → Conv1d → AdaptiveMaxPool1d(1)`.
- Quick non-NN baseline: n-gram counts (`CountVectorizer(analyzer="char", ngram_range=(1,3))`) + LogisticRegression or LightGBM. It is often enough.
- For paths (x,y sequences), hand-built features (length, displacement, turn angles, speed stats) + RandomForest won 2025 L4.

## Time series forecasting
- First try lag and rolling features + LightGBM.
- NN: a window of the last `L` steps → MLP or 1D-CNN → next `H` steps (multi-output regression).
- Normalize per series (subtract the last value or the mean) and add it back afterwards.

## Images (pixel grids, small pictures)
```python
class SmallCNN(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        def block(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                              nn.ReLU(), nn.Conv2d(o, o, 3, padding=1),
                                              nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(block(in_ch, 32), block(32, 64), block(64, 128),
                               nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(128, n_classes)
    def forward(self, x): return self.head(self.f(x))   # x: (B, C, H, W) scaled to ~[0,1]
```
- Augment with flips, small shifts and rotations (`torchvision.transforms.v2`) **if** the label is invariant to them.
- For bigger or real photos, use a pretrained `torchvision.models.resnet18(weights="DEFAULT")` and replace `.fc`. **Download the weights before the contest** (on-site download rule):
  ```bash
  uv run python -c "import torchvision; torchvision.models.resnet18(weights='DEFAULT')"
  ```
- Tiny images (≤ 16×16) work fine flattened into an MLP, or with PCA + boosting.

## Audio
- Load: `y, sr = librosa.load(path, sr=16000)` or `soundfile.read`.
- Features: log-mel spectrogram `librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64))`, then treat it as a 1-channel image and use `SmallCNN`.
- Quick baseline: MFCC mean and std per clip (`librosa.feature.mfcc`) + RandomForest or LightGBM.
- Pad or crop clips to a fixed length.

## Autoencoders / anomaly detection
- Train an MLP autoencoder (`n → 64 → 8 → 64 → n`) on normal data. Reconstruction error is the anomaly score.
- Baselines: `IsolationForest`, or PCA reconstruction error.

## Text (natural language)
- TF-IDF + LogisticRegression is a strong, instant baseline.
- Only use transformers if the weights are pre-downloaded and the data warrants it.
