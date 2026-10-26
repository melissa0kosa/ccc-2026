"""Scratch: (1) v2's critical loss positions re-scored by v6 (raw v, prior, c100, c400, Q400) next to v2;
(2) referee-noise sampling odds of v6's warmup moves in the two v6 losses (60 seeds, 100 sims, noise + temp);
(3) lowmat raw-value calibration in v6's live games (v6 vs v2)."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "reviews"))
from c1 import evaluator, MODELS, mcts
from colony_one.encoding import action_index
from analyze_matches import parse
from sg_pos import env_at
import v6_4f_common, v2_common
EV = {n: evaluator(str(MODELS / f"{n}.onnx")) for n in ("az_res4f_v6", "az_res4f_v2")}
V2POS = [(15273, 26), (15273, 34), (15273, 35), (16438, 26), (16438, 30), (15778, 17), (15778, 20), (15778, 25), (16003, 1), (16211, 3), (15780, 3), (16438, 3)]

def run(env, ev, sims, noise=False, seed=0):
    return mcts.run(env.board, env.current_team, env.actions_left, env.round_number, ev,
                    mcts.MCTSConfig(simulations=sims, add_root_noise=noise), rng=np.random.default_rng(seed))

def setup(m, k):
    mv, _ = parse(m["protocol"]); env = env_at(mv, k)
    name = {action_index(a, env.current_team): env.format_action(a).replace(" ", "") for a in env.legal_actions()}
    played = f"{mv[k][2]}->{mv[k][3]}"; pid = [i for i, n in name.items() if n == played][0]
    return mv, env, name, played, pid

print("=== (1) v2 critical positions, v6 vs v2 (no noise)")
v2m = {m["match_id"]: m for m in v2_common.load()}
for mid, k in V2POS:
    mv, env, name, played, pid = setup(v2m[mid], k)
    idx = np.array(list(name.keys()))
    out = [f"{mid} a{k+1} {env.current_team[:3]} r{env.round_number} played(v2) {mv[k][1]} {played}"]
    for mn, ev in EV.items():
        lg, v = ev(env.observation()[None])
        p = np.exp(lg[0][idx] - lg[0][idx].max()); p /= p.sum(); pr = dict(zip(idx.tolist(), p.tolist()))
        r1 = run(env, ev, 100); b1 = mcts.select_action_index(r1, 0.0)
        r4 = run(env, ev, 400); b4 = mcts.select_action_index(r4, 0.0); N4 = sum(r4.N.values())
        q = lambda a: r4.W[a] / r4.N[a] if r4.N.get(a) else float("nan")
        top = int(idx[np.argmax(p)])
        out.append(f"   {mn:12s} v={v[0]:+.2f} top {name[top]} {p.max():.2f} p(v2played)={pr[pid]:.2f} | c100 {name[b1]} | c400 {name[b4]} Q400(c400)={q(b4):+.2f} Q400(c100)={q(b1):+.2f} Q400(v2played)={q(pid):+.2f} root400 {sum(r4.W.values())/N4:+.2f}")
    print("\n".join(out), flush=True)

print("=== (2) v6 warmup sampling odds (referee: noise on, temp 1.0 for k<4, 0.5 for k<8), 60 seeds")
v6m = {m["match_id"]: m for m in v6_4f_common.load()}
ev = EV["az_res4f_v6"]
for mid, k in [(20222, 2), (20222, 3), (20258, 0), (20258, 1), (20258, 4)]:
    mv, env, name, played, pid = setup(v6m[mid], k)
    temp = 1.0 if k < 4 else 0.5; hits = 0
    for s in range(60):
        rng = np.random.default_rng(1000 + s)
        r = mcts.run(env.board, env.current_team, env.actions_left, env.round_number, ev, mcts.MCTSConfig(simulations=100, add_root_noise=True), rng=rng)
        hits += mcts.select_action_index(r, temp, rng) == pid
    print(f"{mid} a{k+1} played {mv[k][1]} {played} temp {temp}: sampled {hits}/60", flush=True)

print("=== (3) lowmat (each side K + <=1 piece) raw value in v6 live games")
rows = []
for m in v6_4f_common.load():
    mv, _ = parse(m["protocol"]); win = "CYAN" if "CYAN WINS" in m["protocol"] else "MAGENTA"
    first = None
    for k in range(len(mv)):
        env = env_at(mv, k)
        t = [x for l in env.render().splitlines()[:8] for x in l.split()[1:] if x[0] in "cm"]
        nc = sum(1 for x in t if x[0] == "c" and x != "cK"); nm = sum(1 for x in t if x[0] == "m" and x != "mK")
        if nc <= 1 and nm <= 1:
            if first is None:
                us = m["you"][0].lower(); first = (k, sorted(x for x in t if x[0] == us), sorted(x for x in t if x[0] != us))
            z = 1.0 if env.current_team == win else -1.0; o = env.observation()[None]
            rows.append((m["match_id"], z, *[float(e(o)[1][0]) for e in EV.values()]))
    if first:
        print(f"  {m['match_id']} {m['you'][:3]} {'WON ' if win == m['you'] else 'LOST'} lowmat from a{first[0]+1} us {first[1]} them {first[2]}")
    else:
        print(f"  {m['match_id']} {m['you'][:3]} {'WON ' if win == m['you'] else 'LOST'} no lowmat")
if rows:
    R = np.array([r[1:] for r in rows]); z = R[:, 0]
    for i, n in enumerate(EV):
        print(f"  {n}: n={len(rows)} MSE {np.mean((R[:, 1+i]-z)**2):.3f} sign-acc {np.mean(np.sign(R[:, 1+i]) == z):.2f}")
