"""Scratch: critical positions of az_res4f_v2 losses: raw v, prior, MCTS-100/800 for v2, v20, v3 (models/az_res4f_v3.onnx as of now)."""
import sys
from multiprocessing import get_context
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "reviews"))
POS = [(15273, 26), (15273, 34), (15273, 35), (16438, 26), (16438, 27), (16438, 30), (15778, 17), (15778, 20), (15778, 24), (15778, 25), (16003, 1), (16211, 3)]
MODELS_ = ["az_res4f_v2", "az_res_v20", "az_res4f_v3"]

def probe(case):
    from c1 import evaluator, MODELS, mcts
    from colony_one.encoding import action_index
    from analyze_matches import parse
    from sg_pos import env_at
    from v2_common import load
    mid, k = case
    m = next(x for x in load() if x["match_id"] == mid)
    mv, _ = parse(m["protocol"]); env = env_at(mv, k)
    acts = env.legal_actions()
    name = {action_index(a, env.current_team): env.format_action(a).replace(" ", "") for a in acts}
    idx = np.array(list(name.keys()))
    played = f"{mv[k][2]}->{mv[k][3]}"; pid = [i for i, n in name.items() if n == played][0]
    out = [f"{mid} a{k+1} {env.current_team[:3]} r{env.round_number} played {mv[k][1]} {played}"]
    for mn in MODELS_:
        ev = evaluator(str(MODELS / f"{mn}.onnx"))
        lg, v = ev(env.observation()[None])
        p = np.exp(lg[0][idx] - lg[0][idx].max()); p /= p.sum(); pr = dict(zip(idx.tolist(), p.tolist()))
        res = {}
        for sims in (100, 800):
            r = mcts.run(env.board, env.current_team, env.actions_left, env.round_number, ev, mcts.MCTSConfig(simulations=sims, add_root_noise=False), rng=np.random.default_rng(0))
            b = mcts.select_action_index(r, 0.0); N = sum(r.N.values())
            q = lambda a: r.W[a] / r.N[a] if r.N.get(a) else float("nan")
            res[sims] = (name[b], q(b), q(pid), sum(r.W.values()) / N)
        top = int(idx[np.argmax(p)])
        out.append(f"   {mn:12s} v={v[0]:+.2f} top {name[top]} {p.max():.2f} p(played)={pr[pid]:.2f} | c100 {res[100][0]} root100 {res[100][3]:+.2f} | c800 {res[800][0]} Q800={res[800][1]:+.2f} Q800(pl)={res[800][2]:+.2f} root800 {res[800][3]:+.2f}")
    return "\n".join(out)

if __name__ == "__main__":
    with get_context("spawn").Pool(2) as pool:
        for s in pool.map(probe, POS, chunksize=1):
            print(s, flush=True)
