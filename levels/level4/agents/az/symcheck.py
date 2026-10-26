"""Self-check: D4 symmetries map legal moves of a position to legal moves of the transformed position."""
import pickle, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from c1 import CYAN, MAGENTA, encoding
from colony_one.board import Board


def perms():
    P = np.zeros((8, 64), np.int64)
    for s in range(8):
        for sq in range(64):
            r, c = divmod(sq, 8)
            if s & 4:
                r, c = c, r
            if s & 1:
                c = 7 - c
            if s & 2:
                r = 7 - r
            P[s, sq] = r * 8 + c
    return P


PERM = perms()
IDXMAP = (PERM[:, :, None] * 64 + PERM[:, None, :]).reshape(8, 4096)


def tf_obs(obs, s):
    flat = obs.reshape(16, 64)
    out = np.zeros_like(flat)
    out[:, PERM[s]] = flat
    return out.reshape(16, 8, 8)


if __name__ == "__main__":
    f = sorted((HERE / "data").glob("games_teacher_*.pkl"))[0]
    games = pickle.load(open(f, "rb"))
    obs = np.concatenate([g["obs"] for g in games[:150]]).astype(np.float32) / 8
    bad = 0
    tot = 0
    for o in obs[::3]:
        team = MAGENTA if o[15, 0, 0] > 0.5 else CYAN
        m = encoding.legal_action_mask(Board.from_observation(o, team), team)
        for s in range(8):
            ms = encoding.legal_action_mask(Board.from_observation(tf_obs(o, s), team), team)
            exp = np.zeros(4096, bool)
            exp[IDXMAP[s][m]] = True
            tot += 1
            if not np.array_equal(ms, exp):
                bad += 1
    print(f"positions x syms checked {tot}, mismatches {bad}")
