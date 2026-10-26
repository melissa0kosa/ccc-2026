"""Live-loss positions (seed, action) + 10 positions before + 5 random perturbations each, deep teacher labels,
written as sp_v4f-format records to data/endgame_live.pkl.  uv run python levels/level4/agents/distill/livepos2.py 2"""
from __future__ import annotations

import json
import pickle
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
L4 = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(L4))
SEEDS = {"12278": 43, "13136": 22, "13365": 25, "11686": 36, "11791": 27, "12113": 26}
FILES = ["matches_az_res_v20_a.json", "matches_az_res4f_v2_a.json", "matches_az_res_v20_b.json",
         "matches_az_res_v20_final.json"]
NPERT = 5
T = 0.4


def collect():
    from analyze_matches import parse, replay
    found, items = set(), []
    rng = np.random.default_rng(0)
    for f in FILES:
        for m in json.loads((L4 / "data" / "matches" / f).read_text()):
            mid = str(m.get("match_id"))
            if mid not in SEEDS or mid in found or "protocol" not in m:
                continue
            moves, _ = parse(m["protocol"])
            a = SEEDS[mid]
            for act in range(max(1, a - 10), min(a, len(moves)) + 1):
                env = replay(moves[:act - 1])
                if env.terminated:
                    break
                items.append((mid, act, 0, env.clone()))
                for k in range(NPERT):
                    pe = perturb(env, rng)
                    if pe is not None:
                        items.append((mid, act, k + 1, pe))
            found.add(mid)
    print("found seeds", sorted(found), "items", len(items), flush=True)
    return items


def perturb(env, rng):
    """Move one random piece to a random empty open square nearby (or drop a non-kernel piece)."""
    from teacher import KERNEL
    from colony_one.rules import legal_actions_for_team
    e = env.clone()
    b = e.board
    pieces = [(r, c) for r in range(8) for c in range(8) if b.grid[r][c] is not None]
    for _ in range(20):
        r, c = pieces[int(rng.integers(len(pieces)))]
        p = b.grid[r][c]
        if p.piece_type is not KERNEL and rng.random() < 0.15:
            b.grid[r][c] = None
            break
        dr, dc = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
        rr, cc = r + dr, c + dc
        if (dr or dc) and 0 <= rr < 8 and 0 <= cc < 8 and b.grid[rr][cc] is None and not b.locked[rr][cc]:
            b.grid[rr][cc] = p
            b.grid[r][c] = None
            break
    else:
        return None
    if not legal_actions_for_team(b, e.current_team):
        return None
    return e


def label(item):
    mid, act, k, env = item
    from teacher import Searcher, encoding, KERNEL
    S = Searcher()
    team, al, rnd = env.current_team, env.actions_left, env.round_number
    cnt = {}
    for row in env.board.grid:
        for p in row:
            if p is not None and p.piece_type is not KERNEL:
                cnt[p.team] = cnt.get(p.team, 0) + 1
    mx = max(cnt.values()) if cnt else 0
    depth = al + (6 if mx <= 1 else 4)
    acts, sc, best = S.root(env.board, team, al, rnd, depth=depth, margin=2.0)
    obs = encoding.encode_board(env.board, team, al, rnd)
    idx = np.array([encoding.action_index(a, team) for a in acts], np.int16)
    return mid, act, k, obs, idx, np.array(sc, np.float64), float(best), depth


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    items = collect()
    with get_context("spawn").Pool(workers) as pool:
        res = pool.map(label, items, chunksize=2)
    recs, meta = [], []
    for mid, act, k, obs, idx, sc, best, depth in res:
        s = np.clip(sc, -30, 30)
        p = np.exp((s - s.max()) / T)
        p /= p.sum()
        cnt = np.round(p * 200).astype(np.int16)
        cnt[np.argmax(p)] = max(cnt[np.argmax(p)], 1)
        keep = cnt > 0
        q = float(np.tanh(best / 3.0))
        z = float(np.sign(best)) if abs(best) >= 50 else q
        recs.append((obs[:13].astype(np.uint8), obs[13:16, 0, 0].astype(np.float32), z, idx[keep], cnt[keep], q))
        meta.append((mid, act, k, round(best, 2), depth))
    out = HERE / "data" / "endgame_live.pkl"
    pickle.dump(recs, open(out, "wb"))
    print("saved", len(recs), "records to", out, flush=True)
    for m in meta:
        if m[2] == 0:
            print("  seed", m[0], "action", m[1], "teacher best", m[3], "depth", m[4])
