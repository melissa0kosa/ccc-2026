"""SPRINT-3 label generator: champion self-play under referee config (100 sims, warmup noise/temp),
label ~every 2nd-3rd action (most opening actions) with an 800-sim no-noise search by the champion.

uv run python levels/level4/agents/sprint3/gen.py <seed> <minutes> [model] [deep]
Appends one pickle per game to agents/sprint3/data/lab_<seed>.pkl. Stops on time or data/STOP.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import numpy as np  # noqa: E402
from c1 import MODELS, evaluator, mcts, ColonyOneEnv  # noqa: E402


def main():
    seed, minutes = int(sys.argv[1]), float(sys.argv[2])
    model = sys.argv[3] if len(sys.argv) > 3 else str(MODELS / "az_res4f_v6.onnx")
    deep = int(sys.argv[4]) if len(sys.argv) > 4 else 800
    ev = evaluator(str(Path(model).resolve()))
    cfg_play = mcts.MCTSConfig(simulations=100)
    cfg_deep = mcts.MCTSConfig(simulations=deep, add_root_noise=False)
    rng = np.random.default_rng(seed)
    out = HERE / "data" / f"lab_{seed}.pkl"
    stop = HERE / "data" / "STOP"
    t_end = time.time() + minutes * 60
    ng = npos = 0
    t0 = time.time()
    while time.time() < t_end and not stop.exists():
        env = ColonyOneEnv()
        recs = []
        while not env.terminated:
            team, al, rnd = env.current_team, env.actions_left, env.round_number
            mv = mcts.move_number(team, al, rnd)
            plab = 0.75 if mv <= 8 else 0.4
            if rng.random() < plab and time.time() < t_end:
                obs = env.observation()
                root = mcts.run(env.board, team, al, rnd, ev, cfg_deep)
                idx, cnt = mcts.visit_distribution(root)
                if idx:
                    qa = [root.W[a] / max(1, root.N[a]) for a in idx]
                    rq = sum(root.W.values()) / max(1, sum(root.N.values()))
                    recs.append(dict(obs=np.rint(obs * 8).astype(np.uint8), idx=np.array(idx, np.int16),
                                     cnt=np.array(cnt, np.int16), qa=np.array(qa, np.float32), rq=float(rq),
                                     team=team, move=mv))
            act = mcts.search_action(env.board, team, al, rnd, ev, cfg_play, rng=rng)
            env.step(act)
        for r in recs:
            r["z"] = 1.0 if env.winner == r["team"] else -1.0
        with open(out, "ab") as f:
            pickle.dump(dict(seed=seed, game=ng, recs=recs, winner=env.winner), f)
        ng += 1
        npos += len(recs)
        print(f"[{seed}] game {ng} labels {len(recs)} total {npos} {time.time() - t0:.0f}s "
              f"({(time.time() - t0) / max(1, npos):.2f}s/label)", flush=True)


if __name__ == "__main__":
    main()
