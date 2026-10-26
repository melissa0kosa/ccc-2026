"""Scratch: referee warmup sampling odds (noise + temperature) at live positions, current vs sharpened prior."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "reviews"))
from c1 import evaluator, mcts, MODELS
from analyze_matches import parse
from sg_pos import env_at
HV = evaluator(str(MODELS / "hand_v2.onnx"))
def sharp(k):
    return lambda x: (lambda o: (o[0] * k, o[1]))(HV(x))
ms = {m["match_id"]: m for m in json.loads((HERE / "data/matches/matches_hand_v2_1.json").read_text())}
for mid, k, good in [(6177, 6, "G5->F6"), (6509, 1, "E4->G5"), (6509, 5, "A1->A2"), (6032, 15, "F5->F7")]:
    moves, _ = parse(ms[mid]["protocol"]); env = env_at(moves, k)
    played = f"{moves[k][2]}->{moves[k][3]}"
    for kk in (1.0, 1.5, 2.0):
        ev = HV if kk == 1.0 else sharp(kk)
        cnt = {}
        for s in range(40):
            a = mcts.search_action(env.board, env.current_team, env.actions_left, env.round_number, ev,
                                   mcts.MCTSConfig(simulations=100), rng=np.random.default_rng(1000 + s))
            n = env.format_action(a).replace(" ", ""); cnt[n] = cnt.get(n, 0) + 1
        top = sorted(cnt.items(), key=lambda t: -t[1])[:4]
        print(f"{mid} a{k+1} logits x{kk}: P(played {played})={cnt.get(played,0)/40:.2f} P({good})={cnt.get(good,0)/40:.2f} top {top}", flush=True)
    if k >= 8:
        root = mcts.run(env.board, env.current_team, env.actions_left, env.round_number, HV, mcts.MCTSConfig(simulations=100, add_root_noise=False), rng=np.random.default_rng(0))
        from colony_one.encoding import action_index
        nm = {action_index(a, env.current_team): env.format_action(a).replace(" ", "") for a in env.legal_actions()}
        print("   greedy-100 visits:", sorted(((nm[a], n, round(root.W[a] / n, 3)) for a, n in root.N.items() if n), key=lambda t: -t[1])[:6])
