"""Self-check: legal moves of the transformed position == transformed legal moves (all 8 symmetries)."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from teacher import ColonyOneEnv, CYAN, encoding  # noqa
from colony_one.board import Board
from colony_one.rules import legal_actions_for_team
ops = [lambda r, c: (r, c), lambda r, c: (c, 7 - r), lambda r, c: (7 - r, 7 - c), lambda r, c: (7 - c, r),
       lambda r, c: (c, r), lambda r, c: (7 - c, 7 - r), lambda r, c: (r, 7 - c), lambda r, c: (7 - r, c)]
d = np.load(HERE / "data" / "r5_w0.npz")
bad = 0
for i in range(0, 400, 7):
    obs = d["obs"][i].astype(np.float32)
    b = Board.from_observation(obs, CYAN)  # canonical view as cyan
    leg = {(a.from_pos, a.to_pos) for a in legal_actions_for_team(b, CYAN)}
    for o, f in enumerate(ops):
        o2 = np.zeros_like(obs)
        for r in range(8):
            for c in range(8):
                rr, cc = f(r, c)
                o2[:, rr, cc] = obs[:, r, c]
        b2 = Board.from_observation(o2, CYAN)
        leg2 = {(a.from_pos, a.to_pos) for a in legal_actions_for_team(b2, CYAN)}
        tr = {(f(*fp), f(*tp)) for fp, tp in leg}
        if tr != leg2:
            bad += 1
print("mismatches", bad)
