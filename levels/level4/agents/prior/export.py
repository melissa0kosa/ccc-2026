"""Export a trained .pt to models/<name>.onnx, then report graph size, value parity with hand_v3,
held-out agreement (ONNX, referee evaluator) and latency.

    uv run python levels/level4/agents/prior/export.py <name> [--noexport]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, load_games, games_to_arrays, load_fast, PriorNet  # noqa: E402
from c1 import export_onnx, MODELS, evaluator  # noqa: E402

name = sys.argv[1]
ck = torch.load(HERE / f"{name}.pt", weights_only=False)
a = ck["args"]
if a.get("base", "hand") == "4f":
    from common_4f import load_4f, Prior4f
    model = Prior4f(load_4f(), ch=a["ch"], blocks=a["blocks"])
    REF = "az_res4f_v2"
elif a.get("base", "hand") == "az":
    from common_az import load_az, AzPrior
    model = AzPrior(load_az(), ch=a["ch"], blocks=a["blocks"])
    REF = "az_res_v6"
else:
    model = PriorNet(load_fast(), ch=a["ch"], blocks=a["blocks"])
    REF = "hand_v3"
model.load_state_dict(ck["state"])
out = MODELS / f"{name}.onnx"
if "--novalidate" in sys.argv:
    export_onnx(model, out, check=False)
    print("exported (not validated)", out)
elif "--noexport" not in sys.argv:
    for k in range(5):
        try:
            export_onnx(model, out)
            print("exported + ACCEPTED", out)
            break
        except RuntimeError as e:
            print("validate failed, retry:", str(e)[-200:])
    else:
        sys.exit("validator never accepted")

import onnx  # noqa: E402
m = onnx.load(str(out))
print(f"graph: {len(m.graph.node)} nodes, {len(m.graph.initializer)} initializers, "
      f"{out.stat().st_size / 1024:.0f} KB")

games = load_games(sorted((HERE / "data").glob(a["glob"])))
val_g = [g for g in games if (g["seed"] * 7919 + g["game"]) % 10 == 0]
X, T, M, MV, B = games_to_arrays(val_g)
ev_new = evaluator(str(out)); ev_old = evaluator(str(MODELS / f"{REF}.onnx"))
pn, vn, po, vo = [], [], [], []
for i in range(0, len(X), 8):
    p, v = ev_new(X[i:i + 8]); pn.append(np.asarray(p)); vn.append(np.asarray(v))
    p, v = ev_old(X[i:i + 8]); po.append(np.asarray(p)); vo.append(np.asarray(v))
pn = np.concatenate(pn); po = np.concatenate(po); vn = np.concatenate(vn); vo = np.concatenate(vo)
print(f"value max|new-{REF}| = {np.abs(vn - vo).max():.2e} over {len(X)} held-out positions")


def stats(P):
    P = P.copy(); P[~M] = -1e9
    top = P.argmax(1); ag = top == B
    pr = np.exp(P - P.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    t1 = pr.max(1)
    ce = -(T * np.log(np.clip(pr, 1e-12, 1))).sum(1)
    w = MV <= 8
    return dict(agree=float(ag.mean()), agree_warm=float(ag[w].mean()), agree_post=float(ag[~w].mean()),
                ce=float(ce.mean()), med_top1=float(np.median(t1)), med_top1_warm=float(np.median(t1[w])))


so, sn = stats(po), stats(pn)
print(f"held-out {REF}:", json.dumps(so))
print(f"held-out {name}:", json.dumps(sn))
x8 = X[:8]
for e, nm in ((ev_old, REF), (ev_new, name)):
    e(x8); t = time.perf_counter()
    for _ in range(200):
        e(x8)
    print(f"latency {nm}: {(time.perf_counter() - t) / 200 * 1000:.2f} ms/call b8")
with open(HERE / "eval_log.jsonl", "a") as f:
    f.write(json.dumps(dict(name=name, nodes=len(m.graph.node), inits=len(m.graph.initializer),
                            nval=len(X), ref=REF, ref_stats=so, new=sn,
                            vdiff=float(np.abs(vn - vo).max()))) + "\n")

if "--mcts" in sys.argv:
    # referee-relevant: does a 100-sim greedy search (no noise) pick the 800-sim best move?
    from c1 import mcts, CYAN, MAGENTA  # noqa: E402
    from colony_one.board import Board  # noqa: E402
    recs = [r for g in val_g for r in g["recs"] if len(r["idx"]) > 0]
    cfg = mcts.MCTSConfig(simulations=100, add_root_noise=False)
    res = {}
    for e, nm in ((ev_old, REF), (ev_new, name)):
        hit = hitw = nw = 0
        t = time.process_time()
        for r in recs:
            board = Board.from_observation(r["obs"], r["team"])
            root = mcts.run(board, r["team"], r["al"], r["rnd"], e, cfg)
            a = mcts.select_action_index(root, 0.0)
            b = int(r["idx"][int(np.argmax(r["cnt"]))])
            hit += a == b
            if r["move"] <= 8:
                nw += 1; hitw += a == b
        res[nm] = dict(mcts100_agree=hit / len(recs), mcts100_agree_warm=hitw / max(1, nw),
                       ms_per_search=(time.process_time() - t) / len(recs) * 1000)
        print(f"MCTS-100 agreement with 800-sim best, {nm}: {hit / len(recs):.3f} "
              f"(warmup {hitw / max(1, nw):.3f}) n={len(recs)}, {res[nm]['ms_per_search']:.1f} ms cpu/search",
              flush=True)
    with open(HERE / "eval_log.jsonl", "a") as f:
        f.write(json.dumps(dict(name=name, mcts=res)) + "\n")
