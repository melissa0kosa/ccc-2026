"""Gate B: per-side CPU (process_time inside select) for cand vs ref, c1 agents, 100 sims.
usage: gateB_cpu.py ref.onnx games cand.onnx [...]  -> per-side s/game, ms/action, 52-action extrapolation."""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from c1 import CYAN, ColonyOneEnv, MODELS, make_agent  # noqa: E402

ref, games, cands = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
for cand in cands:
    t0 = time.process_time()
    make_agent(str((MODELS / cand).resolve()), 100, 1)
    load = time.process_time() - t0
    side = {cand: [], ref: []}
    per = {cand: [], ref: []}
    for g in range(games):
        A = make_agent(str((MODELS / cand).resolve()), 100, 1000 + 2 * g)
        B = make_agent(str((MODELS / ref).resolve()), 100, 1001 + 2 * g)
        cy, mg = (A, B) if g % 2 == 0 else (B, A)
        env = ColonyOneEnv()
        tot = {id(A): 0.0, id(B): 0.0}
        while not env.terminated:
            ag = cy if env.current_team == CYAN else mg
            t = time.process_time()
            act = ag.select(env.board, env.current_team, env.actions_left, env.round_number)
            dt = time.process_time() - t
            tot[id(ag)] += dt
            per[cand if ag is A else ref].append(dt)
            env.step(act)
        side[cand].append(tot[id(A)])
        side[ref].append(tot[id(B)])
    for n in (cand, ref):
        ms = np.mean(per[n]) * 1e3
        print(f"{n:22s} side cpu/game mean {np.mean(side[n]):.2f}s max {np.max(side[n]):.2f}s | "
              f"ms/action mean {ms:.1f} p95 {np.percentile(per[n], 95) * 1e3:.1f} max {np.max(per[n]) * 1e3:.1f} | "
              f"52-action est {52 * ms / 1e3:.2f}s (p95 {52 * np.percentile(per[n], 95):.2f}s)"
              + (f" | load {load:.2f}s" if n == cand else ""), flush=True)
