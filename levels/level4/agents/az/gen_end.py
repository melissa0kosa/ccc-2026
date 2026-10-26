"""uv run python levels/level4/agents/az/gen_end.py MODEL GAMES TAG [workers]  -> data/games_end_<tag>.pkl"""
if __name__ == "__main__":
    import pickle
    import sys
    import time
    from multiprocessing import get_context
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np

    from sp_end import endgame_game

    model, n, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    seed = int(time.time()) % 100000 * 1000 + 31
    t = time.time()
    with get_context("spawn").Pool(workers) as pool:
        games = [g for g in pool.imap_unordered(endgame_game, [(model, 100, seed + i) for i in range(n)], chunksize=8)
                 if g is not None]
    out = Path(__file__).resolve().parent / "data" / f"games_end_{tag}.pkl"
    pickle.dump(games, open(out, "wb"))
    print(f"gen_end {tag}: {len(games)} games {sum(g['n'] for g in games)} samples "
          f"cyan_won {np.mean([g['cyan_won'] for g in games]):.2f} len {np.mean([g['n'] for g in games]):.1f} "
          f"cpu/game {np.mean([g['cpu'] for g in games]):.2f} {time.time() - t:.0f}s", flush=True)
