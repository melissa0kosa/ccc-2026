"""Overfitting check: policy CE and value MSE of models on held-out self-play games
(sprint2's fresh v6 games, generated after the candidates were trained).

uv run python levels/level4/heldout_check.py model_a.onnx model_b.onnx [n_games]
"""

import glob
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1 import evaluator  # noqa: E402

DATA = Path(__file__).parent / "agents" / "sprint2" / "data"


def load(n_games):
    games = []
    for f in sorted(glob.glob(str(DATA / "*.pkl"))):
        games += pickle.load(open(f, "rb"))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(games), size=min(n_games, len(games)), replace=False)
    obs, tgt, z = [], [], []
    for gi in pick:
        g = games[gi]
        for t in range(len(g["z"])):
            obs.append(np.asarray(g["obs"][t], np.float32))
            tgt.append((np.asarray(g["idx"][t], np.int64), np.asarray(g["p"][t], np.float64)))
            z.append(float(g["z"][t]))
    return np.stack(obs), tgt, np.array(z)


def losses(path, obs, tgt, z):
    ev = evaluator(str(Path(path).resolve()))
    logits, values = [], []
    for i in range(0, len(obs), 256):
        lg, v = ev(obs[i:i + 256])
        logits.append(lg)
        values.append(v)
    logits, values = np.concatenate(logits), np.concatenate(values)
    ce, top1 = [], []
    for row, (idx, p) in zip(logits, tgt):
        sub = row[idx].astype(np.float64)
        sub -= sub.max()
        logp = sub - np.log(np.exp(sub).sum())
        p = p / p.sum()
        ce.append(-(p * logp).sum())
        top1.append(np.argmax(sub) == np.argmax(p))
    return float(np.mean(ce)), float(np.mean((values - z) ** 2)), float(np.mean(top1))


def main():
    a, b = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    obs, tgt, z = load(n)
    print(f"held-out positions: {len(obs)} from {n} fresh v6 games")
    for m in (a, b):
        ce, mse, t1 = losses(m, obs, tgt, z)
        print(f"  {Path(m).name:24s} policy CE {ce:.4f}   value MSE {mse:.4f}   top-1 vs visits {t1:.3f}")


if __name__ == "__main__":
    main()
