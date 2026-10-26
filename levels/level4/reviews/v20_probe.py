"""Scratch: critical positions of az_res_v20's NEW losses (12278, 13136, 13365), probed with v20, hand_v4f, az_res4f_v1, distill_v5hf.
Per model: raw v (side to move), prior top-3, p(played), root800, Q800(played), c100 / c800 with their Q800. k = actions already applied."""
import sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "reviews"))
MODELS_ = ["az_res_v20.onnx", "hand_v4f.onnx", "az_res4f_v1.onnx", "distill_v5hf.onnx"]
POS = [(12278, 38), (12278, 39), (12278, 42), (12278, 50), (12278, 51), (12278, 52),
       (13136, 18), (13136, 34), (13136, 35), (13136, 36), (13136, 38),
       (13365, 20), (13365, 21), (13365, 24), (13365, 60), (13365, 61)]
if len(sys.argv) > 1:
    POS = [tuple(int(y) for y in x.split(":")) for x in sys.argv[1:]]

def run(args):
    mid, k = args
    from c1 import evaluator, MODELS, mcts
    from colony_one.encoding import action_index
    from analyze_matches import parse
    from sg_pos import env_at
    from v20_common import load
    m = next(x for x in load() if x["match_id"] == mid)
    moves, _ = parse(m["protocol"])
    env = env_at(moves, k)
    acts = env.legal_actions()
    name = {action_index(a, env.current_team): env.format_action(a).replace(" ", "") for a in acts}
    played = f"{moves[k][2]}->{moves[k][3]}"
    pidx = [i for i, n in name.items() if n == played][0]
    b = env.render().splitlines()[:8]; t = [x for l in b for x in l.split()[1:] if x[0] in "cm"]
    out = [f"=== {mid} a{k+1} {env.current_team[:3]} (we={m['you'][:3]}) r{env.round_number} al{env.actions_left} played {moves[k][1]} {played}  pieces C{sorted(x[1:] for x in t if x[0]=='c')} M{sorted(x[1:] for x in t if x[0]=='m')}"]
    for mn in MODELS_:
        ev = evaluator(str(MODELS / mn))
        lg, v = ev(env.observation()[None])
        idx = np.array(list(name.keys())); p = np.exp(lg[0][idx] - lg[0][idx].max()); p /= p.sum()
        pr = dict(zip(idx.tolist(), p.tolist())); order = idx[np.argsort(-p)][:3]
        rs = {}
        for s in (100, 800):
            rs[s] = mcts.run(env.board, env.current_team, env.actions_left, env.round_number, ev,
                             mcts.MCTSConfig(simulations=s, add_root_noise=False), rng=np.random.default_rng(0))
        b1 = mcts.select_action_index(rs[100], 0.0); b8 = mcts.select_action_index(rs[800], 0.0); r8 = rs[800]
        q = lambda a: r8.W.get(a, 0) / r8.N[a] if r8.N.get(a) else float("nan")
        out.append(f"  {mn[:-5]:12s} v={v[0]:+.2f} root800={sum(r8.W.values())/sum(r8.N.values()):+.2f} prior top: " + ",".join(f"{name[int(i)]}:{pr[int(i)]:.2f}" for i in order)
                   + f" | p(pl)={pr[pidx]:.2f} Q800(pl)={q(pidx):+.2f} | c100 {name[b1]} ({q(b1):+.2f}) c800 {name[b8]} ({q(b8):+.2f})")
    return "\n".join(out)

if __name__ == "__main__":
    t = time.time()
    with get_context("spawn").Pool(2) as pool:
        res = pool.map(run, POS, chunksize=1)
    txt = "\n".join(res) + f"\nsecs {time.time()-t:.0f}"
    tag = "" if len(sys.argv) == 1 else "_extra"
    (HERE / "reviews" / f"v20_probe{tag}.txt").write_text(txt); print(txt)
