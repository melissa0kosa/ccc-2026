"""Endgame-start self-play (torch-free worker): random Kernel + 0-1 other piece per side on a shrunk board.

Value targets = game outcome only (q is set equal to z so any Q-mixing is a no-op for these samples).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
L4 = HERE.parents[1]
if str(L4) not in sys.path:
    sys.path.insert(0, str(L4))

from c1 import CYAN, MAGENTA, ColonyOneEnv, evaluator, mcts  # noqa: E402
from colony_one.board import Board  # noqa: E402

KERNEL = 5


def random_start(rng, mode="k01"):
    """mode k01: K + 0-1 piece per side, rounds 9-24.  kt1: K+Trace vs K+1 piece, rounds 6-16.
    k2: K + 2 pieces per side (pre-trade), rounds 6-16. Returns (board, team_to_move, round_number)."""
    if mode == "k01":
        if rng.random() < 0.8:
            rnd = int(rng.integers(9, 17))
        else:
            rnd = int(rng.integers(17, 25))
    else:
        rnd = int(rng.integers(6, 17))
    nlock = 0 if rnd <= 8 else (1 if rnd <= 16 else 2)
    obs = np.zeros((16, 8, 8), np.float32)
    open_sq = []
    for r in range(8):
        for c in range(8):
            if min(r, c, 7 - r, 7 - c) < nlock:
                obs[12, r, c] = 1.0
            else:
                open_sq.append((r, c))
    used = set()

    def take(near=None):
        cands = [q for q in open_sq if q not in used]
        if near is not None:
            adj = [q for q in cands if max(abs(q[0] - near[0]), abs(q[1] - near[1])) == 1]
            if adj:
                cands = adj
        sq = cands[int(rng.integers(len(cands)))]
        used.add(sq)
        return sq

    trace_side = int(rng.integers(0, 2))
    for side in (0, 1):                     # 0 = CYAN planes 0-5, 1 = MAGENTA planes 6-11
        k = take()
        obs[side * 6 + KERNEL, k[0], k[1]] = 1.0
        if mode == "k01":
            types = [int(rng.integers(0, 5))] if rng.random() < 0.8 else []
        elif mode == "kt1":
            types = [3] if side == trace_side else [int(rng.integers(0, 5))]
        else:
            types = list(rng.choice(5, size=2, replace=False))
        for t in types:
            sq = take(near=k if rng.random() < 0.6 else None)
            obs[side * 6 + int(t), sq[0], sq[1]] = 1.0
    board = Board.from_observation(obs, CYAN)
    team = CYAN if rng.random() < 0.5 else MAGENTA
    return board, team, rnd


def endgame_game(args):
    model, sims, seed = args[:3]
    start = args[3] if len(args) > 3 else "k01"
    rng = np.random.default_rng(seed)
    ev = evaluator(str(model))
    cfg = mcts.MCTSConfig(simulations=sims, warmup_temperatures=(1.0,), warmup_step_actions=1000)
    env = ColonyOneEnv()
    if isinstance(start, str):
        board, team, rnd = random_start(rng, start)
        al = 2
    else:
        o, team, al, rnd = start
        board = Board.from_observation(o, team)
    env.board, env.current_team, env.actions_left, env.round_number = board, team, al, rnd
    env.action = 200  # past any warmup logic; we pick temperature below
    obs, idxs, probs, teams = [], [], [], []
    t0 = time.process_time()
    try:
        if not env.legal_actions():
            return None
        while not env.terminated:
            team = env.current_team
            root = mcts.run(env.board, team, env.actions_left, env.round_number, ev, cfg, rng=rng)
            ind, cnt = mcts.visit_distribution(root)
            if not ind:
                return None
            cnt = np.asarray(cnt, np.float32)
            obs.append(np.rint(env.observation() * 8).astype(np.uint8))
            idxs.append(np.asarray(ind, np.int16))
            probs.append(cnt / cnt.sum())
            teams.append(team)
            a = mcts.select_action_index(root, 0.3 if len(teams) > 2 else 1.0, rng=rng)
            env.step(a)
    except Exception as e:  # noqa: BLE001
        print("endgame error", repr(e), flush=True)
        return None
    if not teams:
        return None
    w = env.winner
    z = np.array([1.0 if t == w else -1.0 for t in teams], np.float32)
    return dict(obs=np.stack(obs), idx=idxs, p=probs, z=z, q=z.copy(), cyan_won=(w == CYAN),
                n=len(teams), cpu=time.process_time() - t0, endgame=True)
