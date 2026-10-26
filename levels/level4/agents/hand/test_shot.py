import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, torch, torch.nn.functional as F
import fast5
from c1 import ColonyOneEnv, encoding
from colony_one.rules import legal_actions_for_team
net = fast5.FastNet5({}, 0.0, None).eval()
rng = np.random.default_rng(1); bad = tot = nshot = 0
for g in range(60):
    env = ColonyOneEnv()
    while not env.terminated:
        team = env.current_team
        x = torch.tensor(encoding.encode_board(env.board, team, env.actions_left, env.round_number)[None])
        with torch.no_grad():
            base = F.conv2d(x, net.w_base, net.b_base); empty = base[:, 0:1]
            x0 = F.conv2d(x, net.w_sel); r1 = F.conv2d(x0, net.w_shift, padding=1, groups=80)
            r2 = F.conv2d(r1 * empty, net.w_shift, padding=1, groups=80); r3 = F.conv2d(r2 * empty, net.w_shift, padding=1, groups=80)
            sl = F.conv2d(torch.cat([r1, r2, r3], 1), net.w_comb5); jk = F.conv2d(x, net.w_jk, padding=2)
            R = F.conv2d(torch.cat([sl, jk], 1), net.w_reorg)
        shot_own = R[0, 28].numpy() > 0
        A_own = R[0, 0].numpy() > 0
        sh = np.zeros((8, 8), bool); cap = np.zeros((8, 8), bool)
        for a in legal_actions_for_team(env.board, team):
            p = env.board.piece_at(*a.to_pos)
            if p is not None and p.team != team:
                r, c = encoding.canonical_pos(a.to_pos, team)
                cap[r, c] = True
                if a.kind == "shoot":
                    sh[r, c] = True
        enocc = x[0, 6:12].sum(0).numpy() > 0
        tot += 1; nshot += sh.sum()
        if ((shot_own & enocc) != sh).any() or ((A_own & enocc) != cap).any():
            bad += 1
        acts = env.legal_actions(); env.step(acts[rng.integers(len(acts))])
print("positions", tot, "mismatches", bad, "shots seen", nshot)
