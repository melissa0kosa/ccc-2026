"""Endgame/pre-trade self-play starts.
uv run python levels/level4/agents/az/gen_end2.py MODEL TAG N_K01 N_KT1 N_K2 LIVE_REPEATS [workers]
LIVE: every position in the az_res_v20 live games with <=2 non-Kernel pieces per side, LIVE_REPEATS playouts each."""
if __name__ == "__main__":
    import pickle
    import sys
    import time
    from multiprocessing import get_context
    from pathlib import Path

    HERE = Path(__file__).resolve().parent
    L4 = HERE.parents[1]
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(L4)); sys.path.insert(0, str(L4 / "reviews"))
    import numpy as np

    from sp_end import endgame_game

    model, tag = sys.argv[1], sys.argv[2]
    n01, nt1, nk2, rep = (int(a) for a in sys.argv[3:7])
    workers = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    seed = int(time.time()) % 100000 * 1000 + 53
    jobs = [(model, 100, seed + i, "k01") for i in range(n01)]
    jobs += [(model, 100, seed + 100000 + i, "kt1") for i in range(nt1)]
    jobs += [(model, 100, seed + 200000 + i, "k2") for i in range(nk2)]
    live = 0
    if rep > 0:
        from analyze_matches import parse
        from sg_pos import env_at
        from v20_common import load
        for m in load():
            mv, _ = parse(m["protocol"])
            for k in range(len(mv)):
                env = env_at(mv, k)
                if env.terminated:
                    continue
                o = env.observation()
                if o[0:5].sum() <= 2 and o[6:11].sum() <= 2:
                    live += 1
                    for r in range(rep):
                        jobs.append((model, 100, seed + 300000 + 10 * len(jobs) + r,
                                     (o, env.current_team, env.actions_left, env.round_number)))
    t = time.time()
    with get_context("spawn").Pool(workers) as pool:
        games = [g for g in pool.imap_unordered(endgame_game, jobs, chunksize=8) if g is not None]
    out = HERE / "data" / f"games_end_{tag}.pkl"
    pickle.dump(games, open(out, "wb"))
    print(f"gen_end2 {tag}: live start positions {live} x{rep}; {len(games)} games {sum(g['n'] for g in games)} samples "
          f"cyan_won {np.mean([g['cyan_won'] for g in games]):.2f} len {np.mean([g['n'] for g in games]):.1f} "
          f"{time.time() - t:.0f}s", flush=True)
