"""Arena with end-cause stats. usage: myarena.py A B games [workers] [sims]"""
import sys, time, secrets
from collections import Counter
from multiprocessing import get_context
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np


def game(args):
    a, b, g, sims, seed = args
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from c1 import make_agent, ColonyOneEnv, CYAN
    from colony_one.pieces import PieceType
    a_cyan = g % 2 == 0
    A, B = make_agent(a, sims, seed + 2 * g), make_agent(b, sims, seed + 2 * g + 1)
    cy, mg = (A, B) if a_cyan else (B, A)
    env = ColonyOneEnv()
    t = time.process_time()
    cause = "?"
    while not env.terminated:
        ag = cy if env.current_team == CYAN else mg
        act = ag.select(env.board, env.current_team, env.actions_left, env.round_number)
        tgt = env.board.piece_at(*act.to_pos)
        rnd = env.round_number
        env.step(act)
        if env.terminated:
            if tgt is not None and tgt.piece_type is PieceType.KERNEL and tgt.team != env.winner:
                cause = "capture"
            elif env.board.only_kernels_remain():
                cause = "bare"
            else:
                cause = f"fw{rnd}"
    a_won = (env.winner == CYAN) == a_cyan
    return a_won, a_cyan, env.action, cause, time.process_time() - t


def run(a, b, games, workers=4, sims=100, seed=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from c1 import wilson
    seed = secrets.randbits(30) if seed is None else seed
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        res = pool.map(game, [(a, b, g, sims, seed) for g in range(games)], chunksize=1)
    w = sum(r[0] for r in res)
    lo, hi = wilson(w, games)
    cy = [r[0] for r in res if r[1]]; mg = [r[0] for r in res if not r[1]]
    wins = Counter(r[3] for r in res if r[0]); losses = Counter(r[3] for r in res if not r[0])
    print(f"{Path(a).name} vs {Path(b).name}: {w}/{games} = {w/games:.0%} [{lo:.0%},{hi:.0%}] | "
          f"cyan {np.mean(cy):.0%} mag {np.mean(mg):.0%} | {np.mean([r[2] for r in res]):.0f} act, "
          f"{np.mean([r[4] for r in res]):.1f}s cpu/game | wins {dict(wins)} losses {dict(losses)} | "
          f"{time.time()-t0:.0f}s", flush=True)
    import hashlib, json, datetime
    def sha(f):
        try:
            return hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16]
        except OSError:
            return f
    rec = dict(t=datetime.datetime.utcnow().strftime("%H:%M:%S"), a=Path(a).name, b=Path(b).name, sha_a=sha(a), sha_b=sha(b),
               games=games, wins=w, lo=round(lo, 3), hi=round(hi, 3), cyan=float(np.mean(cy)), mag=float(np.mean(mg)), seed=seed)
    with open(Path(__file__).resolve().parent / "data" / "arena_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + chr(10))
    return w, games


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    sims = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    run(a, b, games, workers, sims)
