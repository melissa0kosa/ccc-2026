"""Gate B: TTA exactness + latency. usage: gateB_tta.py base.onnx tta.onnx:syms [...]  (syms like 0,1)
Equivariance: f(T_s x)[:, IDX[s]] == f(x) for s in the TTA group; reference: mean of base logits over the group,
value = tanh(mean(atanh(v))). Positions from real greedy games (encode_board, canonical team view)."""
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from c1 import CYAN, ColonyOneEnv, MODELS, encoding, GreedyAgent, evaluator  # noqa: E402


def perms():
    P = np.zeros((8, 64), np.int64)
    for s in range(8):
        for sq in range(64):
            r, c = divmod(sq, 8)
            if s & 4:
                r, c = c, r
            if s & 1:
                c = 7 - c
            if s & 2:
                r = 7 - r
            P[s, sq] = r * 8 + c
    return P


PERM = perms()
IDX = (PERM[:, :, None] * 64 + PERM[:, None, :]).reshape(8, 4096)


def tf(x, s):  # out[:, PERM[s][sq]] = in[:, sq]
    f = x.reshape(len(x), 16, 64)
    o = np.zeros_like(f)
    o[:, :, PERM[s]] = f
    return o.reshape(x.shape)


X = []
for g in range(8):
    env = ColonyOneEnv()
    ag = [GreedyAgent(seed=g), GreedyAgent(seed=g + 99)]
    while not env.terminated:
        X.append(encoding.encode_board(env.board, env.current_team, env.actions_left, env.round_number))
        env.step((ag[0] if env.current_team == CYAN else ag[1]).select(env.board, env.current_team,
                                                                         env.actions_left, env.round_number))
X = np.stack(X).astype(np.float32)
rng = np.random.default_rng(0)
X = X[rng.choice(len(X), min(96, len(X)), replace=False)]
so = ort.SessionOptions()
so.intra_op_num_threads = 1
base = ort.InferenceSession(str(MODELS / sys.argv[1]), so, providers=["CPUExecutionProvider"])
print(f"{len(X)} positions")
names = [sys.argv[1]]
for arg in sys.argv[2:]:
    name, syms = arg.split(":")
    syms = [int(s) for s in syms.split(",")]
    names.append(name)
    B = ort.InferenceSession(str(MODELS / name), so, providers=["CPUExecutionProvider"])
    pb, vb = B.run(None, {"observation": X})
    ps, vs = [], []
    for s in syms:
        p, v = base.run(None, {"observation": tf(X, s)})
        ps.append(p[:, IDX[s]])
        vs.append(np.arctanh(np.clip(v, -0.999999, 0.999999)))
    pr, vr = np.mean(ps, 0), np.tanh(np.mean(vs, 0))
    print(f"{name} vs reference avg of base over {syms}: max|dp| {np.abs(pb - pr).max():.2e} max|dv| {np.abs(vb - vr).max():.2e}")
    p0, v0 = base.run(None, {"observation": X})
    print(f"  base vs tta: max|dp| {np.abs(pb - p0).max():.3f} max|dv| {np.abs(vb - v0).max():.3f} "
          f"argmax agree {np.mean(pb.argmax(1) == p0.argmax(1)):.2f}")
    for s in range(1, 8):
        p2, v2 = B.run(None, {"observation": tf(X, s)})
        dp, dv = np.abs(p2[:, IDX[s]] - pb).max(), np.abs(v2 - vb).max()
        print(f"  sym {s} {'IN ' if s in syms else 'out'} group: equivariance max|dp| {dp:.2e} max|dv| {dv:.2e}")
# latency (referee evaluator), interleaved
evs = {n: evaluator(str((MODELS / n).resolve())) for n in names}
for n in names:
    evs[n](X[:8])
t = {n: {8: [], 16: [], 1: []} for n in names}
for it in range(120):
    for b in (16, 8, 1):
        i = (it * 8) % (len(X) - 16)
        for n in names:
            s = time.perf_counter()
            evs[n](X[i:i + b])
            t[n][b].append(time.perf_counter() - s)
for n in names:
    m = {b: np.median(t[n][b]) * 1e3 for b in (16, 8, 1)}
    r = {b: m[b] / (np.median(t[names[0]][b]) * 1e3) for b in (16, 8, 1)}
    print(f"{n:22s} b16 {m[16]:6.2f} ms ({r[16]:.2f}x) b8 {m[8]:6.2f} ms ({r[8]:.2f}x) b1 {m[1]:6.2f} ms ({r[1]:.2f}x)")
