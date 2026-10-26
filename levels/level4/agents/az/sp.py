"""Self-play worker (no torch import: keeps worker RAM small)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
L4 = HERE.parents[1]
if str(L4) not in sys.path:
    sys.path.insert(0, str(L4))

from c1 import CYAN, ColonyOneEnv, evaluator, mcts  # noqa: E402

# noise on the root for the whole game (warmup window = whole game); we pick temperature ourselves
CFG_CACHE: dict = {}


def cfg(sims: int):
    if sims not in CFG_CACHE:
        CFG_CACHE[sims] = mcts.MCTSConfig(simulations=sims, warmup_temperatures=(1.0,),
                                          warmup_step_actions=1000)
    return CFG_CACHE[sims]


def selfplay_game(args):
    """Returns dict with obs (n,16,8,8) uint8 (x8), legal idx list, probs list, z (n,), q (n,), team."""
    if len(args) == 5:
        model, sims, seed, temp_moves, late_temp = args
        cyan_m = mag_m = model
    else:
        cyan_m, mag_m, sims, seed, temp_moves, late_temp = args
    rng = np.random.default_rng(seed)
    evs = {CYAN: evaluator(str(cyan_m)), "_m": evaluator(str(mag_m))}
    c = cfg(sims)
    env = ColonyOneEnv()
    obs, idxs, probs, teams, qs = [], [], [], [], []
    t0 = time.process_time()
    try:
        while not env.terminated:
            team = env.current_team
            ev = evs[CYAN] if team == CYAN else evs["_m"]
            root = mcts.run(env.board, team, env.actions_left, env.round_number, ev, c, rng=rng)
            ind, cnt = mcts.visit_distribution(root)
            if not ind:
                return None
            cnt = np.asarray(cnt, np.float32)
            tot = cnt.sum()
            obs.append(np.rint(env.observation() * 8).astype(np.uint8))
            idxs.append(np.asarray(ind, np.int16))
            probs.append((cnt / tot).astype(np.float32))
            teams.append(team)
            qs.append(sum(root.W.values()) / max(1, sum(root.N.values())))
            temp = 1.0 if env.action <= temp_moves else late_temp
            a = mcts.select_action_index(root, temp, rng=rng)
            env.step(a)
    except Exception as e:  # noqa: BLE001
        print("selfplay error", repr(e), flush=True)
        return None
    w = env.winner
    z = np.array([1.0 if t == w else -1.0 for t in teams], np.float32)
    return dict(obs=np.stack(obs), idx=idxs, p=probs, z=z, q=np.asarray(qs, np.float32),
                cyan_won=(w == CYAN), n=len(teams), cpu=time.process_time() - t0)
