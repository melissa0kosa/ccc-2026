"""Pre-arena screen: referee-100 (noise + warmup temp) oracle-hit rate and Q-loss vs the 800-sim labels.

uv run python levels/level4/agents/sprint3/screen.py <labels.pkl> <reps> <workers> model1.onnx [model2.onnx ...]
"""
from __future__ import annotations

import pickle
import sys
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import numpy as np  # noqa: E402


def load(fn):
    recs = []
    with open(fn, "rb") as f:
        while True:
            try:
                recs += pickle.load(f)["recs"]
            except Exception:  # noqa: BLE001
                break
    return recs


def job(args):
    from c1 import evaluator, mcts
    from colony_one import encoding
    from colony_one.board import Board
    fn, model, lo, hi, reps, seed = args
    recs = load(fn)[lo:hi]
    ev = evaluator(str(Path(model).resolve()))
    cfg = mcts.MCTSConfig(simulations=100)
    rng = np.random.default_rng(seed)
    out = []
    for r in recs:
        team, mv = r["team"], r["move"]
        al = rnd = None
        for rr in range(1, 200):
            for a in (2, 1):
                if mcts.move_number(team, a, rr) == mv:
                    al, rnd = a, rr
                    break
            if al:
                break
        obs = r["obs"].astype(np.float32) / 8.0
        board = Board.from_observation(obs, team)
        idx = list(r["idx"]); qa = r["qa"]; best = idx[int(np.argmax(r["cnt"]))]
        for _ in range(reps):
            act = mcts.search_action(board, team, al, rnd, ev, cfg, rng=rng)
            k = encoding.action_index(act, team)
            q = qa[idx.index(k)] if k in idx else qa.min()
            out.append((k == best, float(qa.max() - q) if False else float(qa[idx.index(best)] - q), mv <= 8))
    return out


def main():
    fn, reps, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    models = sys.argv[4:]
    n = len(load(fn))
    pool = get_context("spawn").Pool(workers)
    chunks = [(i * n // workers, (i + 1) * n // workers) for i in range(workers)]
    for m in models:
        res = []
        for o in pool.map(job, [(fn, m, lo, hi, reps, 1000 + i) for i, (lo, hi) in enumerate(chunks)]):
            res += o
        a = np.array(res, dtype=np.float64)
        op = a[:, 2] > 0
        print(f"{Path(m).name}: n={n} reps={reps} hit {a[:, 0].mean():.3f} (opening {a[op, 0].mean():.3f}, "
              f"later {a[~op, 0].mean():.3f}) Qloss {a[:, 1].mean():.4f} (opening {a[op, 1].mean():.4f}, "
              f"later {a[~op, 1].mean():.4f})", flush=True)


if __name__ == "__main__":
    main()
