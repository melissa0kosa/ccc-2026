"""Scratch: CPU per game / per call with the referee's OnnxEvaluator (1 thread)."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from c1 import evaluator, MODELS, make_agent, play_game, ColonyOneEnv
for name in ("stopgap_v1", "hand_v2"):
    p = str(MODELS / f"{name}.onnx"); ev = evaluator(str(Path(p).resolve()))
    X = np.stack([ColonyOneEnv().observation()] * 8)
    ev(X)
    for b in (1, 8):
        c = time.process_time()
        for _ in range(300):
            ev(X[:b])
        print(name, f"batch {b}: {(time.process_time()-c)/300*1e3:.3f} ms cpu/call")
    cpu, n = [], []
    for g in range(6):
        a, bb = make_agent(p, 100, 2 * g), make_agent(p, 100, 2 * g + 1)
        c = time.process_time(); w, k = play_game(a, bb); cpu.append(time.process_time() - c); n.append(k)
    print(name, "self-play cpu/game (both sides):", [round(x, 2) for x in cpu], "actions", n, "mean", round(float(np.mean(cpu)), 2), "max", round(max(cpu), 2))
