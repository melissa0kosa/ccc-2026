"""Scratch: az_res_v6 verification: validator (retry), graph counts, value range, latency 1 thread."""
import sys, time
sys.path.insert(0, "levels/level4"); sys.path.insert(0, "levels/level4/reviews")
import numpy as np, onnx
from c1 import MODELS, validate
from colony_one.onnx_runner import OnnxEvaluator
from colony_one.encoding import action_index
from analyze_matches import parse
from sg_pos import env_at
from v2_common import load
from pathlib import Path; HERE = Path("levels/level4/models"); P = MODELS / "az_res4f_v2.onnx"
m = onnx.load(str(P)); g = m.graph
print("nodes", len(g.node), "initializers", len(g.initializer), "opset", [o.version for o in m.opset_import], "inputs", [(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.input], "outputs", [(o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]) for o in g.output])
import hashlib; print("sha256", hashlib.sha256(P.read_bytes()).hexdigest()[:16], "warm same:", P.read_bytes() == (HERE.parent / "gate" / "az_res4f_v2_warm.onnx").read_bytes())
c = time.process_time(); w = time.time(); ev = OnnxEvaluator(P)
print(f"session create: {1e3*(time.process_time()-c):.0f} ms cpu, {1e3*(time.time()-w):.0f} ms wall")
obs = []
for mm in load():
    mv, _ = parse(mm["protocol"])
    for k in range(len(mv)):
        obs.append(env_at(mv, k).observation())
X = np.stack(obs).astype(np.float32); print("live positions", len(X))
for b in (1, 8):
    ev(X[:b]); c = time.process_time(); n = 0
    for i in range(0, 300 * b, b):
        j = i % (len(X) - b); ev(X[j:j + b]); n += 1
    print(f"batch {b}: {1e3*(time.process_time()-c)/n:.2f} ms cpu/call")
L, V = [], []
for i in range(0, len(X), 8):
    l, v = ev(X[i:i + 8]); L.append(l); V.append(v)
L = np.concatenate(L); V = np.concatenate(V)
print(f"value finite={np.isfinite(V).all()} min={V.min():+.4f} max={V.max():+.4f} |v|>0.9 {np.mean(np.abs(V)>0.9):.1%} mean|v| {np.abs(V).mean():.2f}; logits finite={np.isfinite(L).all()} [{L.min():.2f},{L.max():.2f}]")
for t in range(1):
    ok, out = validate(P)
    print("validate try", t + 1, "ACCEPTED" if ok else "REJECTED"); 
    if ok or "too slow" not in out.lower():
        print(out[-1500:]); break
