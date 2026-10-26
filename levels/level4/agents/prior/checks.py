"""Checks: (1) D4 symmetry of legal moves, (2) torch FastNet(w_v3) == hand_v3.onnx, (3) baseline agreement."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (HERE, IDXMAP, PERM, INV, INVIDX, tf_obs_np, load_games, games_to_arrays,
                    load_fast)  # noqa: E402
from c1 import CYAN, MAGENTA, encoding, MODELS, evaluator  # noqa: E402
from colony_one.board import Board  # noqa: E402

games = load_games(sorted((HERE / "data").glob("gen_*.pkl")))
obs, tgt, mask, move, best = games_to_arrays(games)
print("positions", len(obs))

# (1) symmetry
bad = tot = 0
for o in obs[::2]:
    team = MAGENTA if o[15, 0, 0] > 0.5 else CYAN
    m = encoding.legal_action_mask(Board.from_observation(o, team), team)
    for s in range(8):
        ms = encoding.legal_action_mask(Board.from_observation(tf_obs_np(o, s), team), team)
        exp = np.zeros(4096, bool)
        exp[IDXMAP[s][m]] = True
        # gather form used in training
        assert np.array_equal(m[INVIDX[s]], exp)
        tot += 1
        bad += int(not np.array_equal(ms, exp))
print(f"symmetry: {tot} position x sym checks, {bad} mismatches")

# (2) parity
fn = load_fast().eval()
ev = evaluator(str(MODELS / "hand_v3.onnx"))
with torch.no_grad():
    p1, v1 = fn(torch.from_numpy(obs[:500]))
p2, v2 = ev(obs[:500])
p1 = p1.numpy(); v1 = v1.numpy()
dv = np.abs(v1 - v2).max()
dp = 0.0
for i in range(500):
    a = p1[i][mask[i]]; b = np.asarray(p2[i])[mask[i]]
    dp = max(dp, np.abs((a - a.mean()) - (b - b.mean())).max())
print(f"parity torch vs hand_v3.onnx: max|dv| {dv:.2e}, legal centered logit diff {dp:.2e}")

# (3) baseline agreement
with torch.no_grad():
    P, _ = fn(torch.from_numpy(obs))
P = P.numpy()
P[~mask] = -1e9
top = P.argmax(1)
agree = (top == best).mean()
print(f"base top1 agreement with deep-search best: {agree:.3f} (warmup {(top == best)[move <= 8].mean():.3f})")
