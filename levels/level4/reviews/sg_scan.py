"""Scratch: blunder scan of stopgap_v1 live games with a hand_v2 MCTS(400) reference."""
import json, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "reviews"))

def scan(mid):
    from c1 import evaluator, mcts, MODELS
    from colony_one.encoding import action_index
    from analyze_matches import parse
    from sg_pos import env_at
    SG = evaluator(str(MODELS / "stopgap_v1.onnx")); HV = evaluator(str(MODELS / "hand_v2.onnx"))
    m = next(x for x in json.loads((HERE / "data/matches/matches_1405_stopgap_v1.json").read_text()) if x["match_id"] == mid)
    moves, _ = parse(m["protocol"])
    rows = []
    for k in range(len(moves)):
        if moves[k][0] != m["you"]:
            continue
        env = env_at(moves, k)
        cfg = mcts.MCTSConfig(simulations=400, add_root_noise=False)
        root = mcts.run(env.board, env.current_team, env.actions_left, env.round_number, HV, cfg, rng=np.random.default_rng(0))
        acts = env.legal_actions()
        name = {action_index(a, env.current_team): env.format_action(a).replace(" ", "") for a in acts}
        played = f"{moves[k][2]}->{moves[k][3]}"
        pidx = [i for i, n in name.items() if n == played][0]
        N = sum(root.N.values()); V = sum(root.W.values()) / N
        q = lambda a: root.W.get(a, 0) / root.N[a] if root.N.get(a) else float("nan")
        best = max(root.N, key=root.N.get)
        obs = env.observation()[None]
        rows.append(dict(mid=mid, k=k, team=env.current_team, al=env.actions_left, rnd=env.round_number,
                         played=played, piece=moves[k][1], q_played=q(pidx), n_played=root.N.get(pidx, 0),
                         best=name[best], q_best=q(best), n_best=root.N[best], rootV=V,
                         v_sg=float(SG(obs)[1][0]), v_hv=float(HV(obs)[1][0])))
    return rows

if __name__ == "__main__":
    ms = [m["match_id"] for m in json.loads((HERE / "data/matches/matches_1405_stopgap_v1.json").read_text()) if m["played_at"] >= "2026-10-23T13:52:00"]
    t = time.time()
    with get_context("spawn").Pool(2) as pool:
        res = pool.map(scan, ms, chunksize=1)
    out = [r for rr in res for r in rr]
    (HERE / "reviews" / "sg_scan.json").write_text(json.dumps(out, indent=0))
    for r in out:
        flag = "  <<<" if (r["q_best"] - (r["q_played"] if r["q_played"] == r["q_played"] else -1)) > 0.4 else ""
        print(f"{r['mid']} a{r['k']+1:>2} {r['team'][:3]} r{r['rnd']} al{r['al']} {r['piece']} {r['played']:8s} Qp={r['q_played']:+.2f}(n{r['n_played']:3d}) best {r['best']:8s} Qb={r['q_best']:+.2f}(n{r['n_best']:3d}) root={r['rootV']:+.2f} vSG={r['v_sg']:+.2f} vHV={r['v_hv']:+.2f}{flag}")
    print("secs", time.time() - t)
