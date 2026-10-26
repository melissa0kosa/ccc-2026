"""Exact-parity check for gated wrappers: where do A and B differ at all, and how much material is on the board there?
Positions: greedy/random games + MCTS(A) self-play games (covers real play) + the reviewer's low-material live positions.
usage: parity2.py A.onnx B.onnx [games]"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from c1 import CYAN, ColonyOneEnv, MODELS, encoding, evaluator, GreedyAgent, RandomAgent, SearchAgent  # noqa: E402

a, b = MODELS / sys.argv[1], MODELS / sys.argv[2]
games = int(sys.argv[3]) if len(sys.argv) > 3 else 60
ea, eb = evaluator(str(a.resolve())), evaluator(str(b.resolve()))
X, M, NP = [], [], []


def pieces(env):
    t = [x for ln in env.render().splitlines()[:8] for x in ln.split()[1:] if x[:1] in "cm"]
    nc = sum(1 for x in t if x[0] == "c" and x != "cK")
    nm = sum(1 for x in t if x[0] == "m" and x != "mK")
    return nc, nm


for g in range(games):
    env = ColonyOneEnv()
    if g % 3 == 0:
        ag = [SearchAgent(ea, 50, g), SearchAgent(ea, 50, g + 777)]
    elif g % 3 == 1:
        ag = [GreedyAgent(seed=g), RandomAgent(seed=g)]
    else:
        ag = [GreedyAgent(seed=g), GreedyAgent(seed=g + 1000)]
    while not env.terminated:
        X.append(encoding.encode_board(env.board, env.current_team, env.actions_left, env.round_number))
        M.append(env.legal_mask())
        NP.append(pieces(env))
        agent = ag[0] if env.current_team == CYAN else ag[1]
        env.step(agent.select(env.board, env.current_team, env.actions_left, env.round_number))
X = np.stack(X).astype(np.float32)
M = np.stack(M).astype(bool)
pa, va = ea(X)
pb, vb = eb(X)
dv = np.abs(va - vb)
dp = np.where(M, np.abs(pa - pb), 0).max(1)
diff = (dv > 0) | (dp > 0)
print(f"{a.name} vs {b.name}: {len(X)} positions; differ anywhere: {diff.sum()} ({diff.mean():.1%}); "
      f"max |dv| {dv.max():.3g}, max legal |dlogit| {dp.max():.3g}")
lowmat = np.array([max(n) <= 1 for n in NP])
print(f"  low-material (each side K+<=1): {lowmat.sum()} positions, differing {diff[lowmat].sum()}; "
      f"other positions: {(~lowmat).sum()}, differing {diff[~lowmat].sum()}")
c = Counter(NP[i] for i in np.where(diff)[0])
print("  material (cyan pieces, magenta pieces) at differing positions:", dict(sorted(c.items())[:20]))
