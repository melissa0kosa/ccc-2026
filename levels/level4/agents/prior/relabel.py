"""Relabel saved positions with a deep search of another model (no root noise).

    uv run python levels/level4/agents/prior/relabel.py <tag> <model.onnx> <worker> <nworkers> [deep_sims]
Writes data/lab_<tag>_<worker>.pkl (same record format as gen.py; seeds/game ids kept).
"""
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
from common import load_games  # noqa: E402
from c1 import evaluator, mcts  # noqa: E402
from colony_one.board import Board  # noqa: E402

tag, model, k, nw = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
deep = int(sys.argv[5]) if len(sys.argv) > 5 else 800
games = load_games(sorted((HERE / "data").glob("gen_*.pkl")))
order = np.random.default_rng(123).permutation(len(games))
mine = [games[i] for j, i in enumerate(order) if j % nw == k]
ev = evaluator(str(Path(model).resolve()))
cfg = mcts.MCTSConfig(simulations=deep, add_root_noise=False)
out = HERE / "data" / f"lab_{tag}_{k}.pkl"
stop = HERE / "data" / "STOP_LAB"
t0 = time.time(); n = 0
for gi, g in enumerate(mine):
    if stop.exists():
        break
    recs = []
    for r in g["recs"]:
        board = Board.from_observation(r["obs"], r["team"])
        root = mcts.run(board, r["team"], r["al"], r["rnd"], ev, cfg)
        idx, cnt = mcts.visit_distribution(root)
        q = [root.W[a] / max(1, root.N[a]) for a in idx]
        recs.append(dict(r, idx=np.array(idx, np.int16), cnt=np.array(cnt, np.int16),
                         q=np.array(q, np.float32)))
    with open(out, "ab") as f:
        pickle.dump(dict(seed=g["seed"], game=g["game"], recs=recs, winner=g["winner"]), f)
    n += len(recs)
    print(f"[{k}] game {gi + 1}/{len(mine)} pos {n} {time.time() - t0:.0f}s ({(time.time() - t0) / n:.2f}s/pos)",
          flush=True)
