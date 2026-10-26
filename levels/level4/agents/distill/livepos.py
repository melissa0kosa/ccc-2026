"""Endgame-ish positions from live az_res_v20 matches (replayed), labelled by deep teacher search.
uv run python levels/level4/agents/distill/livepos.py WORKERS"""
from __future__ import annotations

import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
L4 = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(L4))
MAXL = 96
FILES = ["matches_az_res_v20_a.json", "matches_az_res_v20_b.json", "matches_az_res_v20_final.json"]
SEEDS = {"12278": 43, "13136": 22, "13365": 25, "11686": 36, "11791": 27, "12113": 26}


def collect():
    from analyze_matches import parse, replay
    from teacher import KERNEL
    out = []
    for f in FILES:
        for m in json.loads((L4 / "data" / "matches" / f).read_text()):
            if "protocol" not in m:
                continue
            try:
                moves, _ = parse(m["protocol"])
                replay(moves)
            except Exception:
                continue
            mid = str(m["match_id"])
            seed_a = next((a for s, a in SEEDS.items() if s in mid), None)
            for n in range(len(moves)):
                env = replay(moves[:n])
                if env.terminated:
                    break
                cnt = {"CYAN": 0, "MAGENTA": 0}
                for row in env.board.grid:
                    for p in row:
                        if p is not None and p.piece_type is not KERNEL:
                            cnt[p.team] += 1
                sparse = max(cnt.values()) <= 2
                near_seed = seed_a is not None and n >= seed_a - 12
                if sparse or near_seed:
                    out.append((mid, n, env.clone(), max(cnt.values())))
    return out


def label(item):
    mid, n, env, mx = item
    from teacher import Searcher, encoding
    S = Searcher()
    team, al, rnd = env.current_team, env.actions_left, env.round_number
    depth = al + (6 if mx <= 1 else 4)
    acts, sc, best = S.root(env.board, team, al, rnd, depth=depth, margin=2.0)
    obs = encoding.encode_board(env.board, team, al, rnd)
    idx = [encoding.action_index(a, team) for a in acts]
    return mid, n, obs, idx, list(map(float, sc)), float(best)


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    items = collect()
    print("positions", len(items), flush=True)
    with get_context("spawn").Pool(workers) as pool:
        res = pool.map(label, items, chunksize=4)
    O, LI, SC, BEST, Z, NA = [], [], [], [], [], []
    for mid, n, obs, idx, sc, best in res:
        li = np.full(MAXL, -1, np.int16)
        s = np.zeros(MAXL, np.float32)
        k = min(len(idx), MAXL)
        li[:k] = idx[:k]
        s[:k] = sc[:k]
        O.append(obs.astype(np.float16)); LI.append(li); SC.append(s); BEST.append(best)
        Z.append(float(np.tanh(best / 3.0))); NA.append(n)
    np.savez(HERE / "data" / "live_v20.npz", obs=np.stack(O), li=np.stack(LI), sc=np.stack(SC),
             best=np.array(BEST, np.float32), z=np.array(Z, np.float32), act=np.array(NA, np.int16))
    print("saved", len(O), flush=True)
