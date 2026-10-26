"""Add variants to an existing warm_eval run (same positions, refs, seeds); round-1 positions only by default.

    uv run python levels/level4/agents/prior/warm_eval2.py <variant.onnx>...
"""
import json, pickle, sys, time
from multiprocessing import get_context
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
from warm_eval import samp_job  # noqa: E402

if __name__ == "__main__":
    d = pickle.load(open(HERE / "data" / "warm_eval_raw.pkl", "rb"))
    pos, refs, res, V = d["pos"], d["refs"], d["res"], d["variants"]
    n = len(pos)
    sel = [i for i, r in enumerate(pos) if r["rnd"] == 1]
    seeds = list(range(1000, 1010))
    t0 = time.time()
    new = {}
    with get_context("spawn").Pool(3) as pool:
        for v in sys.argv[1:]:
            out = pool.map(samp_job, [(v, pos[i], [s + 7919 * i for s in seeds]) for i in sel], chunksize=4)
            new[v] = dict(zip(sel, out))
    print(f"sampling {time.time() - t0:.0f}s")

    def loss(i, acts):
        ref = refs[i]; best = max(ref, key=lambda a: (ref[a][0], -a)); qmin = min(q for nn, q in ref.values() if nn > 0)
        return np.array([ref[best][1] - (ref[a][1] if a in ref and ref[a][0] > 0 else qmin) for a in acts]), \
            np.array([a == best for a in acts])
    allv = [(Path(v).name, {i: res[vi * n + i] for i in sel}) for vi, v in enumerate(V)] + \
           [(Path(v).name, new[v]) for v in sys.argv[1:]]
    base = np.array([loss(i, res[i])[0].mean() for i in sel])
    summ = {}
    for name, acts in allv:
        L = np.array([loss(i, acts[i])[0] for i in sel]); B = np.array([loss(i, acts[i])[1] for i in sel])
        dpos = L.mean(1) - base
        summ[name] = dict(qloss=float(L.mean()), best_rate=float(B.mean()), gt02=int((L > 0.2).sum()),
                          d_vs_champ=float(dpos.mean()), d_se=float(dpos.std() / np.sqrt(len(dpos))),
                          pos_worse_005=int((dpos > 0.05).sum()), pos_better_005=int((dpos < -0.05).sum()))
        print(name, json.dumps(summ[name]))
    with open(HERE / "warm_eval.jsonl", "a") as f:
        f.write(json.dumps(dict(round1_only=True, n=len(sel), nseeds=10, summary=summ)) + "\n")
