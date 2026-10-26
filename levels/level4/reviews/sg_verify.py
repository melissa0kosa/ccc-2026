"""Scratch: shapes, value range on real live positions, latency, CPU/game for stopgap_v1."""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "reviews"))
from c1 import evaluator, MODELS, make_agent, play_game, CYAN
from analyze_matches import parse
from sg_pos import env_at
import onnxruntime as ort
p = MODELS / "stopgap_v1.onnx"
s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
print("inputs", [(i.name, i.shape, i.type) for i in s.get_inputs()], "outputs", [(o.name, o.shape, o.type) for o in s.get_outputs()])
obs = []
for m in json.loads((HERE / "data/matches/matches_1405_stopgap_v1.json").read_text()):
    mv, _ = parse(m["protocol"])
    for k in range(0, len(mv), 1):
        obs.append(env_at(mv, k).observation())
X = np.stack(obs).astype(np.float32)
pol, val = s.run(None, {"observation": X})
print("positions", X.shape, "policy", pol.shape, pol.dtype, "value", val.shape, val.dtype,
      "value min/max", val.min(), val.max(), "finite", np.isfinite(pol).all() and np.isfinite(val).all())
print("logit range", pol.min(), pol.max())
for b in (1, 8, 32):
    t = time.perf_counter(); c = time.process_time()
    for _ in range(200):
        s.run(None, {"observation": X[:b]})
    print(f"batch {b}: {(time.perf_counter()-t)/200*1e3:.2f} ms wall, {(time.process_time()-c)/200*1e3:.2f} ms cpu per call")
cpu = []
for g in range(4):
    a, b = make_agent(str(p), 100, 2 * g), make_agent(str(p), 100, 2 * g + 1)
    c = time.process_time(); w, n = play_game(a, b); cpu.append(time.process_time() - c)
    print("selfplay game", g, w, n, "actions", f"{cpu[-1]:.2f}s cpu (both sides)")
print("mean cpu/game both sides", np.mean(cpu), "-> per side ~", np.mean(cpu) / 2)
