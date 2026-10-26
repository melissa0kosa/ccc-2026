"""Generate teacher-labelled positions. uv run python levels/level4/agents/distill/gen.py TAG MINUTES WORKERS"""
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
FOLLOW = os.environ.get("FOLLOW", "0") == "1"
KCH = 8


def child_obs(board, team, al, rnd, a):
    from teacher import KERNEL, SHRINK, MAGENTA_, opp, encoding
    rec = board.apply_action(a)
    try:
        cp = rec.captured_piece
        if cp is not None and cp.piece_type is KERNEL:
            return None
        if al > 1:
            return encoding.encode_board(board, team, al - 1, rnd), 1.0
        if board.only_kernels_remain():
            return None
        fw = board.apply_firewall_shrink() if (team == MAGENTA_ and rnd in SHRINK) else None
        try:
            if fw is not None and any(p.piece_type is KERNEL for _, _, p in fw.killed):
                return None
            nr = rnd + 1 if team == MAGENTA_ else rnd
            return encoding.encode_board(board, opp(team), 2, nr), -1.0
        finally:
            if fw is not None:
                board.undo_firewall_shrink(fw)
    finally:
        board.undo_action(rec)


def worker(args):
    tag, wid, minutes, seed, models = args
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from teacher import Searcher, ColonyOneEnv, encoding, CYAN, GreedyAgent, MAGENTA_ as MAG
    rng = np.random.default_rng(seed)
    S = Searcher()
    t_end = time.time() + minutes * 60
    out = HERE / "data" / f"{tag}_w{wid}.npz"
    out.parent.mkdir(exist_ok=True)
    O, LI, SC, BEST, Z, NA = [], [], [], [], [], []
    CO, CV = [], []
    games = 0
    while time.time() < t_end:
        env = ColonyOneEnv()
        mode = rng.random()
        # behaviour per side: 0 teacher, 1 greedy, 2 random
        if mode < 0.6:
            beh = {CYAN: 0, MAG: 0}
        elif mode < 0.8:
            beh = {CYAN: 1, MAG: 0} if rng.random() < .5 else {CYAN: 0, MAG: 1}
        else:
            beh = {CYAN: 2, MAG: 0} if rng.random() < .5 else {CYAN: 0, MAG: 2}
        greedy = GreedyAgent(seed=int(rng.integers(1 << 30)))
        if models:
            from c1 import SearchAgent, evaluator
            ma = models[int(rng.integers(len(models)))]
            mb = models[int(rng.integers(len(models)))]
            agents = {CYAN: SearchAgent(evaluator(ma), 100, int(rng.integers(1 << 30))),
                      MAG: SearchAgent(evaluator(mb), 100, int(rng.integers(1 << 30)))}
            beh = {CYAN: 3, MAG: 3}
        eps = rng.choice([0.0, 0.05, 0.15])
        temp_open = rng.choice([0.3, 0.6, 1.0])
        n_rand_open = int(rng.choice([0, 0, 2, 4]))
        recs = []
        while not env.terminated:
            team = env.current_team
            obs = env.observation()
            acts, sc, best = S.root(env.board, team, env.actions_left, env.round_number)
            idx = [encoding.action_index(a, team) for a in acts]
            li = np.full(MAXL, -1, np.int16)
            s = np.zeros(MAXL, np.float32)
            n = min(len(idx), MAXL)
            li[:n] = idx[:n]
            s[:n] = sc[:n]
            recs.append((obs.astype(np.float16), li, s, best, team, env.action))
            if FOLLOW and env.actions_left == 2:
                # follow-up samples: after the teacher's best quiet first actions, label the second action
                grid = env.board.grid
                quiet = [k for k in np.argsort(-np.array(sc)) if grid[acts[k].to_pos[0]][acts[k].to_pos[1]] is None][:2]
                for k in quiet:
                    rec = env.board.apply_action(acts[k])
                    try:
                        a2, sc2, best2 = S.root(env.board, team, 1, env.round_number)
                        if a2:
                            o2 = encoding.encode_board(env.board, team, 1, env.round_number)
                            idx2 = [encoding.action_index(a, team) for a in a2]
                            li2 = np.full(MAXL, -1, np.int16)
                            s2 = np.zeros(MAXL, np.float32)
                            n2 = min(len(idx2), MAXL)
                            li2[:n2] = idx2[:n2]
                            s2[:n2] = sc2[:n2]
                            recs.append((o2.astype(np.float16), li2, s2, best2, team, env.action))
                    finally:
                        env.board.undo_action(rec)
            for j in rng.permutation(len(acts))[:KCH]:
                co = child_obs(env.board, team, env.actions_left, env.round_number, acts[j])
                if co is not None:
                    CO.append(np.round(co[0] * 8).astype(np.uint8))
                    CV.append(co[1] * sc[j])
            b = beh[team]
            if env.action <= n_rand_open or b == 2 and rng.random() < 0.7 or rng.random() < eps:
                a = acts[int(rng.integers(len(acts)))]
            elif b == 1:
                a = greedy.select(env.board, team, env.actions_left, env.round_number)
            elif b == 3:
                a = agents[team].select(env.board, team, env.actions_left, env.round_number)
            else:
                scs = np.array(sc, np.float64)
                T = temp_open if env.action <= 8 else 0.1
                p = np.exp((scs - scs.max()) / T)
                p /= p.sum()
                a = acts[int(rng.choice(len(acts), p=p))]
            env.step(a)
        for (o, li, s, best, team, act) in recs:
            O.append(o); LI.append(li); SC.append(s); BEST.append(best)
            Z.append(1.0 if env.winner == team else -1.0); NA.append(act)
        games += 1
        if games % 1 == 0:
            np.savez(out, obs=np.stack(O), li=np.stack(LI), sc=np.stack(SC), best=np.array(BEST, np.float32),
                     z=np.array(Z, np.float32), act=np.array(NA, np.int16),
                     ch_obs=np.stack(CO), ch_v=np.array(CV, np.float32))
    np.savez(out, obs=np.stack(O), li=np.stack(LI), sc=np.stack(SC), best=np.array(BEST, np.float32),
             z=np.array(Z, np.float32), act=np.array(NA, np.int16),
                     ch_obs=np.stack(CO), ch_v=np.array(CV, np.float32))
    return games, len(O)


if __name__ == "__main__":
    tag = sys.argv[1]
    minutes = float(sys.argv[2])
    workers = int(sys.argv[3])
    models = [str((HERE.parents[1] / "models" / m).resolve()) for m in sys.argv[4].split(",")] if len(sys.argv) > 4 else []
    base = int(time.time()) % 100000
    with get_context("spawn").Pool(workers) as pool:
        res = pool.map(worker, [(tag, w, minutes, base * 100 + w, models) for w in range(workers)])
    print("games/positions per worker", res, flush=True)
