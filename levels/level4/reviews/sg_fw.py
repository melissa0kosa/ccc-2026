"""Scratch: at CYAN's round-8 turn, play both actions with each model (MCTS100, 4 seeds) and
report the own material left on the outer ring (burns after MAGENTA's turn)."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "reviews"))
from c1 import evaluator, mcts, MODELS
from analyze_matches import parse
from sg_pos import env_at
EVS = {n: evaluator(str(MODELS / f"{n}.onnx")) for n in ("stopgap_v1", "hand_v2")}
ms = {m["match_id"]: m for m in json.loads((HERE / "data/matches/matches_1405_stopgap_v1.json").read_text())}
for mid, k in [(5471, 28), (5079, 28), (5051, 28)]:
    mv, _ = parse(ms[mid]["protocol"])
    for name, ev in EVS.items():
        res = []
        for s in range(4):
            env = env_at(mv, k); seq = []
            team = env.current_team
            while env.current_team == team and not env.terminated:
                a = mcts.search_action(env.board, env.current_team, env.actions_left, env.round_number, ev,
                                       mcts.MCTSConfig(simulations=100), rng=np.random.default_rng(s))
                seq.append(env.format_action(a).replace(" ", "")); env.step(a)
            obs = env.observation()  # MAGENTA to move -> CYAN pieces are planes 6-10
            ring0 = np.zeros((8, 8), bool); ring0[[0, 7], :] = True; ring0[:, [0, 7]] = True
            left = [n for i, n in enumerate("PDXTI") if (obs[6 + i] * ring0).sum() > 0]
            res.append(f"{'+'.join(seq)} ring0:{''.join(left) or '-'}")
        print(mid, name, res)
