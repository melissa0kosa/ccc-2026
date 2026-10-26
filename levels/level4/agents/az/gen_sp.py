"""Generate self-play games with a fixed model (+ some vs an opponent) into data/games_res_<tag>.pkl.
uv run python levels/level4/agents/az/gen_sp.py MODEL OPP SELF_GAMES MIX_GAMES TAG [workers]"""
if __name__ == "__main__":
    import sys
    import time
    from multiprocessing import get_context
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import azlib

    model, opp, n_self, n_mix, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    seed = int(time.time()) % 100000 * 1000 + 777
    t = time.time()
    jobs = azlib.sp_jobs(model, model, n_self, 100, seed, 12, 0.3)
    jobs += azlib.sp_jobs(model, opp, n_mix, 100, seed + n_self, 12, 0.3, alternate=True)
    with get_context("spawn").Pool(workers) as pool:
        games = azlib.run_jobs(pool, jobs, f"res_{tag}")
    azlib.log(f"gen_sp {tag}: {azlib.summarize(games)} {time.time() - t:.0f}s")
