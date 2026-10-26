"""Offline warmup Q-loss: referee config (noise + warmup temperature) per variant vs an 800-sim reference.

    uv run python levels/level4/agents/prior/warm_eval.py <champion.onnx> <n_per_round> <seeds> <variant.onnx>...
"""
import json
import pickle
import sys
import time
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

_EV = {}


def ev(path):
    from c1 import evaluator
    return evaluator(str(Path(path).resolve()))


def ref_job(args):
    from c1 import mcts
    from colony_one.board import Board
    champ, r = args
    board = Board.from_observation(r["obs"], r["team"])
    root = mcts.run(board, r["team"], r["al"], r["rnd"], ev(champ),
                    mcts.MCTSConfig(simulations=800, add_root_noise=False))
    return {a: (root.N[a], root.W[a] / max(1, root.N[a])) for a in root.N}


def samp_job(args):
    from c1 import mcts
    from colony_one.board import Board
    model, r, seeds = args
    out = []
    cfg = mcts.MCTSConfig(simulations=100)
    for s in seeds:
        rng = np.random.default_rng(s)
        board = Board.from_observation(r["obs"], r["team"])
        T = mcts.warmup_temperature(r["move"], cfg.warmup_temperatures, cfg.warmup_step_actions)
        root = mcts.run(board, r["team"], r["al"], r["rnd"], ev(model), cfg, rng=rng)
        out.append(mcts.select_action_index(root, T, rng=rng))
    return out


if __name__ == "__main__":
    from common import load_games
    champ, nper, nseeds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    variants = [champ] + sys.argv[4:]
    games = load_games(sorted((HERE / "data").glob("gen_*.pkl")))
    seen, pool_r = set(), {1: [], 2: []}
    for g in games:
        for r in g["recs"]:
            if r["move"] <= 8 and len(r["idx"]) > 1:
                k = r["obs"].tobytes()
                if k not in seen:
                    seen.add(k); pool_r[r["rnd"]].append(r)
    rng = np.random.default_rng(7)
    pos = []
    for rd in (1, 2):
        p = pool_r[rd]
        pos += [p[i] for i in rng.permutation(len(p))[:nper]]
    print(f"unique warmup positions: r1 {len(pool_r[1])} r2 {len(pool_r[2])}; using {len(pos)}", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(3) as pool:
        refs = pool.map(ref_job, [(champ, r) for r in pos], chunksize=4)
        print(f"refs done {time.time() - t0:.0f}s", flush=True)
        seeds = list(range(1000, 1000 + nseeds))
        jobs = [(v, r, [s + 7919 * i for s in seeds]) for v in variants for i, r in enumerate(pos)]
        res = pool.map(samp_job, jobs, chunksize=4)
    print(f"sampling done {time.time() - t0:.0f}s", flush=True)
    summary = {}
    for vi, v in enumerate(variants):
        L = {1: [], 2: []}; B = {1: [], 2: []}; big = {1: 0, 2: 0}
        for i, r in enumerate(pos):
            ref = refs[i]
            best = max(ref, key=lambda a: (ref[a][0], -a))
            qmin = min(q for n, q in ref.values() if n > 0)
            for a in res[vi * len(pos) + i]:
                q = ref[a][1] if a in ref and ref[a][0] > 0 else qmin
                loss = ref[best][1] - q
                L[r["rnd"]].append(loss); B[r["rnd"]].append(a == best); big[r["rnd"]] += loss > 0.2
        s = {f"r{rd}": dict(qloss=float(np.mean(L[rd])), se=float(np.std(L[rd]) / np.sqrt(len(L[rd]))),
                            best_rate=float(np.mean(B[rd])), n=len(L[rd]), loss_gt_0_2=big[rd])
             for rd in (1, 2)}
        summary[Path(v).name] = s
        print(Path(v).name, json.dumps(s), flush=True)
    with open(HERE / "warm_eval.jsonl", "a") as f:
        f.write(json.dumps(dict(champ=champ, nper=nper, nseeds=nseeds, summary=summary)) + "\n")
    with open(HERE / "data" / "warm_eval_raw.pkl", "wb") as f:
        pickle.dump(dict(pos=pos, refs=refs, res=res, variants=variants), f)
