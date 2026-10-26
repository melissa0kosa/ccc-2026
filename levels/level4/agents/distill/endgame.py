"""Random sparse endgames (Kernel + 0-2 pieces per side), labelled with a DEEP teacher search.
uv run python levels/level4/agents/distill/endgame.py TAG MINUTES WORKERS
Value target stored in z = tanh(best/3) (so the training blend is pure teacher), best = deep score."""
from __future__ import annotations

import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
MAXL = 96
import os
MODE = os.environ.get("EGMODE", "")
KCH = 8


def make_position(rng):
    from teacher import CYAN_, MAGENTA_, KERNEL
    from colony_one.board import Board
    from colony_one.pieces import Piece, PieceType
    b = Board()
    rnd = int(rng.choice([6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 18, 20, 22, 24, 25]))
    b.locked = Board.locked_cells_for_round(rnd)
    for r in range(8):
        for c in range(8):
            b.grid[r][c] = None
    free = [(r, c) for r in range(8) for c in range(8) if not b.locked[r][c]]
    rng.shuffle(free)
    others = [PieceType.PING, PieceType.DAEMON, PieceType.EXPLOIT, PieceType.TRACE, PieceType.ICE]
    k = 0
    for team in (CYAN_, MAGENTA_):
        b.grid[free[k][0]][free[k][1]] = Piece(KERNEL, team)
        k += 1
    if MODE == "KT":
        # Kernel + 1 piece each side (one side's piece is a Trace half the time), rounds 6-16
        rnd = int(rng.integers(6, 17))
        b.locked = Board.locked_cells_for_round(rnd)
        for (r, c) in list(free):
            if b.locked[r][c]:
                free.remove((r, c))
        for r in range(8):
            for c in range(8):
                b.grid[r][c] = None
        k = 0
        for team in (CYAN_, MAGENTA_):
            b.grid[free[k][0]][free[k][1]] = Piece(KERNEL, team)
            k += 1
        tside = CYAN_ if rng.random() < 0.5 else MAGENTA_
        for team in (CYAN_, MAGENTA_):
            if team == tside and rng.random() < 0.6:
                t = 3  # Trace
            else:
                t = int(rng.integers(len(others)))
            if rng.random() < 0.1:
                continue
            b.grid[free[k][0]][free[k][1]] = Piece(others[t], team)
            k += 1
    else:
        for team in (CYAN_, MAGENTA_):
            n = int(rng.choice([0, 1, 1, 1, 2]))
            for t in rng.choice(len(others), size=n, replace=False):
                b.grid[free[k][0]][free[k][1]] = Piece(others[int(t)], team)
                k += 1
    if b.only_kernels_remain() and rng.random() < 0.7:
        return None  # mostly skip bare-kernel positions (decided by rule at turn end)
    team = CYAN_ if rng.random() < 0.5 else MAGENTA_
    al = 2 if rng.random() < 0.6 else 1
    return b, team, al, rnd


def worker(args):
    tag, wid, minutes, seed = args
    from teacher import Searcher, encoding
    from gen import child_obs
    rng = np.random.default_rng(seed)
    S = Searcher()
    t_end = time.time() + minutes * 60
    out = HERE / "data" / f"{tag}_w{wid}.npz"
    O, LI, SC, BEST, Z, NA, CO, CV = [], [], [], [], [], [], [], []
    n = 0
    while time.time() < t_end:
        try:
            pos = make_position(rng)
        except IndexError:
            continue
        if pos is None:
            continue
        b, team, al, rnd = pos
        from colony_one.rules import legal_actions_for_team
        if not legal_actions_for_team(b, team):
            continue
        depth = al + (6 if MODE == "KT" else 4)  # rest of turn + 3 (KT) or 2 full turns
        acts, sc, best = S.root(b, team, al, rnd, depth=depth, margin=2.0)
        if abs(best) >= 50 and rng.random() < 0.75:
            continue  # keep only a quarter of the trivially decided positions
        obs = encoding.encode_board(b, team, al, rnd)
        idx = [encoding.action_index(a, team) for a in acts]
        li = np.full(MAXL, -1, np.int16)
        s = np.zeros(MAXL, np.float32)
        m = min(len(idx), MAXL)
        li[:m] = idx[:m]
        s[:m] = sc[:m]
        O.append(obs.astype(np.float16)); LI.append(li); SC.append(s); BEST.append(best)
        Z.append(float(np.tanh(best / 3.0))); NA.append(0)
        for j in rng.permutation(len(acts))[:KCH]:
            co = child_obs(b, team, al, rnd, acts[j])
            if co is not None:
                CO.append(np.round(co[0] * 8).astype(np.uint8))
                CV.append(co[1] * sc[j])
        n += 1
        if n % 20 == 0:
            np.savez(out, obs=np.stack(O), li=np.stack(LI), sc=np.stack(SC), best=np.array(BEST, np.float32),
                     z=np.array(Z, np.float32), act=np.array(NA, np.int16),
                     ch_obs=np.stack(CO), ch_v=np.array(CV, np.float32))
    np.savez(out, obs=np.stack(O), li=np.stack(LI), sc=np.stack(SC), best=np.array(BEST, np.float32),
             z=np.array(Z, np.float32), act=np.array(NA, np.int16),
             ch_obs=np.stack(CO), ch_v=np.array(CV, np.float32))
    return n


if __name__ == "__main__":
    tag, minutes, workers = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    if workers == 0:  # inline timing test
        import teacher
        rng = np.random.default_rng(0)
        S = teacher.Searcher()
        for i in range(15):
            pos = make_position(rng)
            if pos is None:
                continue
            b, team, al, rnd = pos
            t = time.perf_counter()
            acts, sc, best = S.root(b, team, al, rnd, depth=al + 4, margin=2.0)
            print(f"r{rnd} {team[:1]} al{al} n={len(acts)} best={best:.2f} {time.perf_counter()-t:.2f}s nodes={S.nodes}")
        sys.exit()
    base = int(time.time()) % 100000
    with get_context("spawn").Pool(workers) as pool:
        print(pool.map(worker, [(tag, w, minutes, base * 100 + w) for w in range(workers)]), flush=True)
