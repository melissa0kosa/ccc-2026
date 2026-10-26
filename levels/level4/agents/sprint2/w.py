"""SPRINT-2 self-play worker (torch-free). Normal games or games from a given mid-game start; records visits, z, root Q."""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
L4 = HERE.parents[1]
for p in (str(L4), str(L4 / "agents" / "az")):
    if p not in sys.path:
        sys.path.insert(0, p)
from c1 import CYAN, ColonyOneEnv, evaluator, mcts  # noqa: E402
from colony_one.board import Board  # noqa: E402

_EV = {}
_CFG = {}

def game(args):
    model, sims, seed, start = args
    rng = np.random.default_rng(seed)
    if model not in _EV:
        _EV[model] = evaluator(str(model))
    ev = _EV[model]
    if sims not in _CFG:
        _CFG[sims] = mcts.MCTSConfig(simulations=sims, warmup_temperatures=(1.0,), warmup_step_actions=1000)
    c = _CFG[sims]
    env = ColonyOneEnv()
    temp_moves = 12
    if start is not None:
        o, team, al, rnd = start
        env.board, env.current_team, env.actions_left, env.round_number = Board.from_observation(o, team), team, al, rnd
        env.action = 200
    obs, idxs, probs, teams, qs = [], [], [], [], []
    t0 = time.process_time()
    try:
        if not env.legal_actions():
            return None
        while not env.terminated:
            team = env.current_team
            root = mcts.run(env.board, team, env.actions_left, env.round_number, ev, c, rng=rng)
            ind, cnt = mcts.visit_distribution(root)
            if not ind:
                return None
            cnt = np.asarray(cnt, np.float32)
            obs.append(np.rint(env.observation() * 8).astype(np.uint8))
            idxs.append(np.asarray(ind, np.int16))
            probs.append((cnt / cnt.sum()).astype(np.float32))
            teams.append(team)
            qs.append(sum(root.W.values()) / max(1, sum(root.N.values())))
            if start is None:
                temp = 1.0 if env.action <= temp_moves else 0.3
            else:
                temp = 1.0 if len(teams) <= 2 else 0.3
            env.step(mcts.select_action_index(root, temp, rng=rng))
    except Exception as e:  # noqa: BLE001
        print("sp2 error", repr(e), flush=True)
        return None
    if not teams:
        return None
    w = env.winner
    z = np.array([1.0 if t == w else -1.0 for t in teams], np.float32)
    return dict(obs=np.stack(obs), idx=idxs, p=probs, z=z, q=np.asarray(qs, np.float32),
                cyan_won=(w == CYAN), n=len(teams), cpu=time.process_time() - t0, mid=start is not None)
