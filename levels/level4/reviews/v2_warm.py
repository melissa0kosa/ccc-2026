"""Scratch: warmup blunders in az_res4f_v2 live games: how often does the referee's noisy 100-sim warmup sample the played move?
Referee: dirichlet 0.3/0.15, temp 1.0 for actions 1-4, 0.5 for 5-8, sampling by visits."""
import sys
from multiprocessing import get_context
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "reviews"))
CASES = [(15780, 2), (15780, 3), (16438, 3), (16003, 1), (16211, 2), (16211, 3), (14956, 4), (14998, 3)]
SEEDS = 60

def probe(case):
    from c1 import evaluator, MODELS, mcts
    from colony_one.encoding import action_index
    from analyze_matches import parse
    from sg_pos import env_at
    from v2_common import load
    mid, k = case
    ev = evaluator(str(MODELS / "az_res4f_v2.onnx"))
    m = next(x for x in load() if x["match_id"] == mid)
    mv, _ = parse(m["protocol"]); env = env_at(mv, k)
    name = {action_index(a, env.current_team): env.format_action(a).replace(" ", "") for a in env.legal_actions()}
    played = f"{mv[k][2]}->{mv[k][3]}"
    temp = 1.0 if k < 4 else 0.5
    hits, share = 0, []
    for s in range(SEEDS):
        rng = np.random.default_rng(1000 + s)
        cfg = mcts.MCTSConfig(simulations=100, add_root_noise=True)
        r = mcts.run(env.board, env.current_team, env.actions_left, env.round_number, ev, cfg, rng=rng)
        a = mcts.select_action_index(r, temp, rng)
        N = sum(r.N.values()); pid = [i for i, n in name.items() if n == played][0]
        share.append(r.N.get(pid, 0) / N); hits += name[a] == played
    cfg = mcts.MCTSConfig(simulations=100, add_root_noise=False)
    r = mcts.run(env.board, env.current_team, env.actions_left, env.round_number, ev, cfg, rng=np.random.default_rng(0))
    pid = [i for i, n in name.items() if n == played][0]
    q = lambda a: r.W[a] / r.N[a] if r.N.get(a) else float("nan")
    best = max(r.N, key=r.N.get)
    return f"{mid} a{k+1} {env.current_team[:3]} played {played} temp {temp}: sampled {hits}/{SEEDS} ({hits/SEEDS:.0%}), mean visit share {np.mean(share):.3f} | no-noise c100 {name[best]} Q100={q(best):+.2f}, played N={r.N.get(pid,0)} Q100={q(pid):+.2f}"

if __name__ == "__main__":
    with get_context("spawn").Pool(2) as pool:
        for line in pool.map(probe, CASES, chunksize=1):
            print(line, flush=True)
