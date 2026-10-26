"""Check attack maps against engine legal captures on random positions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from hand import HandNet, DEFAULT_POL, n_features
from c1 import ColonyOneEnv, encoding, CYAN
from colony_one.rules import legal_actions_for_team
rng = np.random.default_rng(0)
net = HandNet(n_features()); net.pp = DEFAULT_POL
bad = 0; tot = 0
for g in range(150):
    env = ColonyOneEnv()
    while not env.terminated:
        team = env.current_team
        x = encoding.encode_board(env.board, team, env.actions_left, env.round_number)
        with torch.no_grad():
            own, en, locked, empty, a_own, a_en = net.maps(torch.tensor(x[None]))
        A = a_own.sum(1)[0].numpy()
        acts = legal_actions_for_team(env.board, team)
        cap = np.zeros((8, 8))
        for a in acts:
            p = env.board.piece_at(*a.to_pos)
            if p is not None and p.team != team:
                r, c = encoding.canonical_pos(a.to_pos, team); cap[r, c] = 1
        enocc = x[6:12].sum(0) > 0
        pred = (A > 0) & enocc
        tot += 1
        if (pred != (cap > 0)).any():
            bad += 1
            if bad < 4:
                print("mismatch", np.argwhere(pred != (cap > 0)), env.board)
        acts = env.legal_actions() if hasattr(env, 'legal_actions') else acts
        env.step(acts[rng.integers(len(acts))])
print("positions", tot, "mismatches", bad)
