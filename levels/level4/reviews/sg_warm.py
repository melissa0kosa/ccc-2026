"""Scratch: 4869 a7 (warmup, T=0.5): probability the referee samples the saving capture G7->F6."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "reviews"))
from c1 import evaluator, mcts, MODELS
from colony_one.encoding import action_index
from analyze_matches import parse
from sg_pos import env_at
m = next(x for x in json.loads((HERE / "data/matches/matches_1405_stopgap_v1.json").read_text()) if x["match_id"] == 4869)
mv, _ = parse(m["protocol"])
for k in (6, 7):
    env = env_at(mv, k)
    T = mcts.warmup_temperature(mcts.move_number(env.current_team, env.actions_left, env.round_number))
    save = [action_index(a, env.current_team) for a in env.legal_actions() if env.format_action(a).replace(" ", "") == "G7->F6"][0]
    for name in ("stopgap_v1", "hand_v2"):
        ev = evaluator(str(MODELS / f"{name}.onnx")); ps, tops = [], []
        for s in range(10):
            root = mcts.run(env.board, env.current_team, env.actions_left, env.round_number, ev, mcts.MCTSConfig(), rng=np.random.default_rng(s))
            N = np.array(list(root.N.values()), float); w = N ** (1 / T)
            ps.append(root.N.get(save, 0) ** (1 / T) / w.sum()); tops.append(root.N.get(save, 0) / N.sum())
        print(f"a{k+1} T={T} {name}: P(sample G7->F6)={np.mean(ps):.2f}  visit share={np.mean(tops):.2f}")
