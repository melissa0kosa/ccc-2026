"""uv run python levels/level4/agents/sprint2/gen.py MODEL SECONDS [workers]  -> sprint2/data/sp_<ts>.pkl (time-bounded)"""
if __name__ == "__main__":
    import json, pickle, sys, time
    from multiprocessing import get_context
    from pathlib import Path
    HERE = Path(__file__).resolve().parent
    L4 = HERE.parents[1]
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(L4)); sys.path.insert(0, str(L4 / "reviews"))
    import numpy as np
    from w import game
    model, secs = sys.argv[1], float(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    t0 = time.time()
    seed = int(time.time()) % 100000 * 1000 + 4242
    rng = np.random.default_rng(seed)
    # mid-game starts from live match protocols
    from analyze_matches import parse
    from c1 import ColonyOneEnv
    starts, seen = [], set()
    for f in sorted((L4 / "data" / "matches").glob("*.json")):
        try:
            ms = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(ms, list):
            continue
        for m in ms:
            if not isinstance(m, dict) or not m.get("protocol") or m.get("match_id") in seen:
                continue
            seen.add(m["match_id"])
            try:
                mv, _ = parse(m["protocol"])
                env = ColonyOneEnv(); pos = []
                for team, _pc, a, b, _k in mv:
                    if env.terminated:
                        break
                    if len(pos) >= 0 and env.action >= 6:
                        pos.append((env.observation().copy(), env.current_team, env.actions_left, env.round_number))
                    c = [x for x in env.legal_actions() if env.format_action(x).replace(" ", "") == f"{a}->{b}"]
                    if len(c) != 1:
                        break
                    env.step(c[0])
                if pos:
                    for k in rng.choice(len(pos), size=min(6, len(pos)), replace=False):
                        starts.append(pos[int(k)])
            except Exception:  # noqa: BLE001
                continue
    print(f"gen: {len(seen)} live games -> {len(starts)} mid-game starts ({time.time()-t0:.0f}s)", flush=True)
    N = 6000
    jobs = []
    si = 0
    for i in range(N):
        if i % 4 == 3 and starts:
            jobs.append((model, 100, seed + i, starts[si % len(starts)])); si += 1
        else:
            jobs.append((model, 100, seed + i, None))
    out = []
    (HERE / "data").mkdir(exist_ok=True)
    dst = HERE / "data" / f"sp_{seed}.pkl"
    pool = get_context("spawn").Pool(workers)
    last = time.time()
    try:
        for r in pool.imap_unordered(game, jobs, chunksize=2):
            if r is not None:
                out.append(r)
            if time.time() - last > 60:
                pickle.dump(out, open(dst, "wb")); last = time.time()
                print(f"gen: {len(out)} games {sum(g['n'] for g in out)} pos {time.time()-t0:.0f}s", flush=True)
            if time.time() - t0 > secs:
                break
    finally:
        pool.terminate()
    pickle.dump(out, open(dst, "wb"))
    mid = [g for g in out if g["mid"]]
    print(f"gen DONE {dst.name}: {len(out)} games ({len(mid)} mid) {sum(g['n'] for g in out)} pos cyan_won "
          f"{np.mean([g['cyan_won'] for g in out]):.2f} cpu/game {np.mean([g['cpu'] for g in out]):.2f} {time.time()-t0:.0f}s", flush=True)
