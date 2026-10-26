"""Relative inference latency (referee OnnxEvaluator, batch 8 and 1, real positions), interleaved so load hits all
models equally. usage: latency.py ref.onnx cand.onnx [...]"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from c1 import CYAN, ColonyOneEnv, MODELS, encoding, evaluator, GreedyAgent  # noqa: E402

names = sys.argv[1:]
evs = {n: evaluator(str((MODELS / n).resolve())) for n in names}
X = []
for g in range(6):
    env = ColonyOneEnv()
    ag = [GreedyAgent(seed=g), GreedyAgent(seed=g + 99)]
    while not env.terminated:
        X.append(encoding.encode_board(env.board, env.current_team, env.actions_left, env.round_number))
        env.step((ag[0] if env.current_team == CYAN else ag[1]).select(env.board, env.current_team,
                                                                         env.actions_left, env.round_number))
X = np.stack(X).astype(np.float32)
t = {n: {8: [], 1: []} for n in names}
for n in names:  # warm-up
    evs[n](X[:8])
for it in range(150):
    for b in (8, 1):
        i = (it * 8) % (len(X) - 8)
        for n in names:
            s = time.perf_counter()
            evs[n](X[i:i + b])
            t[n][b].append(time.perf_counter() - s)
ref = names[0]
for n in names:
    m8, m1 = np.median(t[n][8]) * 1e3, np.median(t[n][1]) * 1e3
    print(f"{n:20s} b8 {m8:6.2f} ms  b1 {m1:6.2f} ms  ratio vs {ref} b8 {m8 / (np.median(t[ref][8]) * 1e3):.2f}")
