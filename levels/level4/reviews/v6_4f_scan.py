"""Scratch: az_res4f_v6 live decisions, 1 process: prior/value, MCTS-100 (verification vs played, greedy, no noise)
and, for lost games, MCTS-400 (Q of played / c100 / c400, root value).
usage: uv run python levels/level4/reviews/v6_4f_scan.py -> reviews/v6_4f_scan.{json,txt}"""
import json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "reviews"))
from c1 import evaluator, MODELS, mcts
from colony_one.encoding import action_index
from analyze_matches import parse
from sg_pos import env_at
from v6_4f_common import load, MODEL

def search(env, ev, sims):
    cfg = mcts.MCTSConfig(simulations=sims, add_root_noise=False)
    return mcts.run(env.board, env.current_team, env.actions_left, env.round_number, ev, cfg, rng=np.random.default_rng(0))

HV = evaluator(str(MODELS / f"{MODEL}.onnx"))
rows = []; t0 = time.time()
for m in load():
    win = "CYAN" if "CYAN WINS" in m["protocol"] else "MAGENTA"; lost = win != m["you"]
    moves, _ = parse(m["protocol"])
    for k in range(len(moves)):
        if moves[k][0] != m["you"]:
            continue
        env = env_at(moves, k)
        acts = env.legal_actions()
        name = {action_index(a, env.current_team): env.format_action(a).replace(" ", "") for a in acts}
        lg, v = HV(env.observation()[None])
        idx = np.array(list(name.keys()))
        p = np.exp(lg[0][idx] - lg[0][idx].max()); p /= p.sum()
        prior = dict(zip(idx.tolist(), p.tolist()))
        top = int(idx[np.argmax(p)])
        played = f"{moves[k][2]}->{moves[k][3]}"
        pidx = [i for i, n in name.items() if n == played][0]
        r1 = search(env, HV, 100); b1 = mcts.select_action_index(r1, 0.0); N1 = sum(r1.N.values())
        row = dict(mid=m["match_id"], lost=lost, k=k, team=env.current_team, al=env.actions_left, rnd=env.round_number,
                   nlegal=len(acts), played=played, piece=moves[k][1], v=float(v[0]), p_top=float(p.max()), top=name[top],
                   p_played=prior[pidx], c100=name[b1], root100=sum(r1.W.values()) / N1)
        if lost:
            r4 = search(env, HV, 400); b4 = mcts.select_action_index(r4, 0.0); N4 = sum(r4.N.values())
            q4 = lambda a: r4.W.get(a, 0) / r4.N[a] if r4.N.get(a) else float("nan")
            row.update(c400=name[b4], q400_c100=q4(b1), q400_c400=q4(b4), q400_played=q4(pidx), root400=sum(r4.W.values()) / N4,
                       p_c400=prior[b4], n400_c400=r4.N[b4] / N4)
        rows.append(row)
    print(m["match_id"], "done", f"{time.time() - t0:.0f}s", flush=True)
(HERE / "reviews" / "v6_4f_scan.json").write_text(json.dumps(rows, indent=0))
lines = []
for r in rows:
    s = (f"{r['mid']} a{r['k']+1:>2} {r['team'][:3]} r{r['rnd']} al{r['al']} n{r['nlegal']:2d} {r['piece']} played {r['played']:7s} p={r['p_played']:.2f}"
         f" | top {r['top']:7s} {r['p_top']:.2f} v={r['v']:+.2f} | c100 {r['c100']:7s} root100={r['root100']:+.2f}")
    if "c400" in r:
        s += f" | c400 {r['c400']:7s} Q400(c100)={r['q400_c100']:+.2f} Q400(c400)={r['q400_c400']:+.2f} Q400(pl)={r['q400_played']:+.2f} root400={r['root400']:+.2f}"
        if r["c100"] != r["c400"]:
            s += f"  DIFF dQ={r['q400_c400'] - r['q400_c100']:+.2f}"
    if r["k"] >= 8 and r["c100"] != r["played"]:
        s += "  !!PLAYED!=LOCAL100"
    lines.append(s)
post = [r for r in rows if r["k"] >= 8]
lines.append(f"VERIFY post-warmup (k>=8): local c100 == played {sum(r['c100'] == r['played'] for r in post)}/{len(post)}; warmup (k<8): {sum(r['c100'] == r['played'] for r in rows if r['k'] < 8)}/{sum(r['k'] < 8 for r in rows)}")
lines.append(f"secs {time.time() - t0:.0f}")
(HERE / "reviews" / "v6_4f_scan.txt").write_text("\n".join(lines))
print("\n".join(lines[-2:]))
