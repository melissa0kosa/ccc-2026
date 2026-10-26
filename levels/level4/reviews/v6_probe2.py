"""Scratch: probe critical positions with several models: raw v, prior, MCTS-100/800 choice, Q800 of played move."""
import json, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "reviews"))
MODELS_ = ["az_res_v6.onnx", "az_res_v20.onnx", "hand_v4f.onnx"]
FILES = ["matches_az_res_v6.json", "matches_az_res_v20_a.json"]
POS = [(11686, 35), (12113, 31), (12113, 29), (11791, 54), (11791, 58), (11791, 46), (11686, 33), (11057, 23)]

def run(args):
    mid, k = args
    from c1 import evaluator, MODELS, mcts
    from colony_one.encoding import action_index
    from analyze_matches import parse
    from sg_pos import env_at
    m = next(x for f in FILES for x in json.loads((HERE / "data/matches" / f).read_text()) if x["match_id"] == mid)
    moves, _ = parse(m["protocol"])
    env = env_at(moves, k)
    acts = env.legal_actions()
    name = {action_index(a, env.current_team): env.format_action(a).replace(" ", "") for a in acts}
    played = f"{moves[k][2]}->{moves[k][3]}"
    pidx = [i for i, n in name.items() if n == played][0]
    out = [f"=== {mid} a{k+1} {env.current_team[:3]} (we={m['you'][:3]}) r{env.round_number} al{env.actions_left} played {played}"]
    for mn in MODELS_:
        ev = evaluator(str(MODELS / mn))
        lg, v = ev(env.observation()[None])
        idx = np.array(list(name.keys())); p = np.exp(lg[0][idx] - lg[0][idx].max()); p /= p.sum()
        pr = dict(zip(idx.tolist(), p.tolist())); order = idx[np.argsort(-p)][:3]
        rs = {}
        for s in (100, 800):
            r = mcts.run(env.board, env.current_team, env.actions_left, env.round_number, ev,
                         mcts.MCTSConfig(simulations=s, add_root_noise=False), rng=np.random.default_rng(0))
            rs[s] = r
        b1 = mcts.select_action_index(rs[100], 0.0); b8 = mcts.select_action_index(rs[800], 0.0); r8 = rs[800]
        q = lambda a: r8.W.get(a, 0) / r8.N[a] if r8.N.get(a) else float("nan")
        out.append(f"  {mn[:-5]:11s} v={v[0]:+.2f} root800={sum(r8.W.values())/sum(r8.N.values()):+.2f} prior top: " + ",".join(f"{name[int(i)]}:{pr[int(i)]:.2f}" for i in order)
                   + f" | p(played)={pr[pidx]:.2f} Q800(played)={q(pidx):+.2f} | c100 {name[b1]} (Q800 {q(b1):+.2f}) c800 {name[b8]} (Q800 {q(b8):+.2f})")
    return "\n".join(out)

if __name__ == "__main__":
    t = time.time()
    with get_context("spawn").Pool(2) as pool:
        res = pool.map(run, POS, chunksize=1)
    txt = "\n".join(res) + f"\nsecs {time.time()-t:.0f}"
    (HERE / "reviews" / "v6_probe2.txt").write_text(txt); print(txt)
