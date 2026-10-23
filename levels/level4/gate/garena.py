"""Gate arena: c1-identical game loop, but streams one JSON line per game (with action-sequence hashes).

usage (repo root):
  uv run python levels/level4/gate/garena.py <log.jsonl> <workers> A:B:games [A:B:games ...]
A/B: model file names in levels/level4/models (or "greedy"). Jobs run interleaved in the given order.
Seeds: fresh secrets.randbits(30) per match; agent seeds seed+2g / seed+2g+1 exactly like c1.arena.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import sys
import time
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
L4 = HERE.parent
sys.path.insert(0, str(L4))
from c1 import CYAN, ColonyOneEnv, make_agent, MODELS, encoding  # noqa: E402


def spec(name: str) -> str:
    return name if name in ("greedy", "random") else str((MODELS / name).resolve())


def game(job):
    mid, a, b, g, sims, seed = job
    a_cyan = g % 2 == 0
    A, B = make_agent(spec(a), sims, seed + 2 * g), make_agent(spec(b), sims, seed + 2 * g + 1)
    cy, mg = (A, B) if a_cyan else (B, A)
    env = ColonyOneEnv()
    seq = []
    hb = ""
    t = time.process_time()
    while not env.terminated:
        ag = cy if env.current_team == CYAN else mg
        act = ag.select(env.board, env.current_team, env.actions_left, env.round_number)
        seq.append(f"{tuple(act.from_pos)}{tuple(act.to_pos)}")
        env.step(act)
        if len(seq) == 8 and not env.terminated:
            hb = hashlib.md5(encoding.encode_board(env.board, env.current_team, env.actions_left,
                                                   env.round_number).tobytes()
                             + env.render().encode()).hexdigest()[:12]
    cpu = time.process_time() - t
    s = "|".join(seq)
    return dict(mid=mid, a=a, b=b, g=g, seed=seed, a_cyan=a_cyan, a_won=(env.winner == CYAN) == a_cyan,
                winner=str(env.winner), n=env.action, cpu=round(cpu, 3),
                h=hashlib.md5(s.encode()).hexdigest()[:12],
                h8=hashlib.md5("|".join(seq[:8]).encode()).hexdigest()[:12], hb=hb,
                hs=hashlib.md5("|".join(seq[8:]).encode()).hexdigest()[:12])


if __name__ == "__main__":
    log = Path(sys.argv[1])
    workers = int(sys.argv[2])
    sims = 100
    matches = []
    for m in sys.argv[3:]:
        a, b, n = m.split(":")
        matches.append((f"{a}~{b}~{secrets.randbits(16):04x}", a, b, int(n), secrets.randbits(30)))
    # interleave: all jobs of match 1, then match 2... (priority order)
    jobs = [(mid, a, b, g, sims, seed) for mid, a, b, n, seed in matches for g in range(n)]
    print(f"{len(jobs)} games, {workers} workers -> {log}", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool, log.open("a") as f:
        for i, r in enumerate(pool.imap(game, jobs, chunksize=1), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"{i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print("done", time.time() - t0, flush=True)
