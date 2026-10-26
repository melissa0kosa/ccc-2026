"""Scratch: az_res_v20 live decisions (model as argv[1]): prior/value, MCTS-100 vs MCTS-800 (greedy, no noise, referee tie-break).
usage: uv run python levels/level4/reviews/v6_scan.py -> reviews/v6_scan.json / v6_scan.txt"""
import json, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "reviews"))
MODEL = sys.argv[1] if len(sys.argv) > 1 else "az_res4f_v2.onnx"
ONLY = [int(x) for x in sys.argv[2:]]

def search(env, ev, sims):
    from c1 import mcts
    cfg = mcts.MCTSConfig(simulations=sims, add_root_noise=False)
    return mcts.run(env.board, env.current_team, env.actions_left, env.round_number, ev, cfg, rng=np.random.default_rng(0))

def scan(mid):
    from c1 import evaluator, MODELS, mcts
    from colony_one.encoding import action_index
    from analyze_matches import parse
    from sg_pos import env_at
    from v2_common import load
    HV = evaluator(str(MODELS / MODEL))
    m = next(x for x in load() if x["match_id"] == mid)
    moves, _ = parse(m["protocol"])
    rows = []
    for k in range(len(moves)):
        if moves[k][0] != m["you"]:
            continue
        env = env_at(moves, k)
        acts = env.legal_actions()
        name = {action_index(a, env.current_team): env.format_action(a).replace(" ", "") for a in acts}
        obs = env.observation()[None]
        lg, v = HV(obs)
        idx = np.array(list(name.keys()))
        p = np.exp(lg[0][idx] - lg[0][idx].max()); p /= p.sum()
        prior = dict(zip(idx.tolist(), p.tolist()))
        top = int(idx[np.argmax(p)])
        played = f"{moves[k][2]}->{moves[k][3]}"
        pidx = [i for i, n in name.items() if n == played][0]
        r1 = search(env, HV, 100); r8 = search(env, HV, 800)
        b1 = mcts.select_action_index(r1, 0.0); b8 = mcts.select_action_index(r8, 0.0)
        q8 = lambda a: r8.W.get(a, 0) / r8.N[a] if r8.N.get(a) else float("nan")
        N8 = sum(r8.N.values()); N1 = sum(r1.N.values())
        rows.append(dict(mid=mid, k=k, team=env.current_team, al=env.actions_left, rnd=env.round_number, nlegal=len(acts),
                         played=played, piece=moves[k][1], v=float(v[0]), p_top=float(p.max()), top=name[top],
                         p_played=prior[pidx], c100=name[b1], c800=name[b8], q800_c100=q8(b1), q800_c800=q8(b8),
                         q800_played=q8(pidx), n800_c800=r8.N[b8] / N8, n100_c100=r1.N[b1] / N1, n100_played=r1.N.get(pidx, 0) / N1,
                         root100=sum(r1.W.values()) / N1, root800=sum(r8.W.values()) / N8,
                         p_c800=prior[b8]))
    return rows

if __name__ == "__main__":
    from v2_common import load
    ms = [m["match_id"] for m in load() if not ONLY or m["match_id"] in ONLY]
    t = time.time()
    with get_context("spawn").Pool(2) as pool:
        res = pool.map(scan, ms, chunksize=1)
    out = [r for rr in res for r in rr]
    (HERE / "reviews" / f"v2_scan_{MODEL[:-5]}.json").write_text(json.dumps(out, indent=0))
    lines = []
    for r in out:
        flag = ""
        if r["c100"] != r["c800"]:
            flag = f"  DIFF dQ={r['q800_c800'] - r['q800_c100']:+.2f}"
        if r["k"] >= 8 and r["c100"] != r["played"]:
            flag += "  !!PLAYED!=LOCAL100"
        lines.append(f"{r['mid']} a{r['k']+1:>2} {r['team'][:3]} r{r['rnd']} al{r['al']} n{r['nlegal']:2d} {r['piece']} played {r['played']:7s} p={r['p_played']:.2f} | top {r['top']:7s} {r['p_top']:.2f} v={r['v']:+.2f} | c100 {r['c100']:7s} c800 {r['c800']:7s} Q800(c100)={r['q800_c100']:+.2f} Q800(c800)={r['q800_c800']:+.2f} Q800(pl)={r['q800_played']:+.2f} root100={r['root100']:+.2f} root800={r['root800']:+.2f}{flag}")
    lines.append(f"secs {time.time() - t:.0f}")
    (HERE / "reviews" / f"v2_scan_{MODEL[:-5]}.txt").write_text("\n".join(lines))
    print("\n".join(lines))
