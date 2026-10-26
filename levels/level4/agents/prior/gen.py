"""Generate deep-search policy targets.

Play games with PLAY model at 100 sims (referee config: warmup noise + sampling), both colours
(self-play), and label every position with a DEEP search (LABEL model, no root noise).

    uv run python levels/level4/agents/prior/gen.py <tag> <seed> <minutes> [play.onnx] [label.onnx] [deep_sims]

Appends one pickle record per game to agents/prior/data/gen_<tag>_<seed>.pkl.
Stops after <minutes> or when agents/prior/data/STOP exists.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import numpy as np  # noqa: E402
from c1 import MODELS, evaluator, mcts, ColonyOneEnv, CYAN  # noqa: E402


def main():
    tag, seed, minutes = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    play = sys.argv[4] if len(sys.argv) > 4 else str(MODELS / "hand_v3.onnx")
    label = sys.argv[5] if len(sys.argv) > 5 else str(MODELS / "hand_v3.onnx")
    deep = int(sys.argv[6]) if len(sys.argv) > 6 else 800
    evp = evaluator(str(Path(play).resolve()))
    evl = evaluator(str(Path(label).resolve()))
    cfg_play = mcts.MCTSConfig(simulations=100)
    cfg_deep = mcts.MCTSConfig(simulations=deep, add_root_noise=False)
    rng = np.random.default_rng(seed)
    out = HERE / "data" / f"gen_{tag}_{seed}.pkl"
    stop = HERE / "data" / "STOP"
    t_end = time.time() + minutes * 60
    ng = npos = 0
    t0 = time.time()
    while time.time() < t_end and not stop.exists():
        env = ColonyOneEnv()
        recs = []
        while not env.terminated:
            team, al, rnd = env.current_team, env.actions_left, env.round_number
            obs = env.observation()
            root = mcts.run(env.board, team, al, rnd, evl, cfg_deep)
            idx, cnt = mcts.visit_distribution(root)
            q = [root.W[a] / max(1, root.N[a]) if hasattr(root, "W") else 0.0 for a in idx]
            recs.append(dict(obs=obs.astype(np.float32), idx=np.array(idx, np.int16),
                             cnt=np.array(cnt, np.int16), q=np.array(q, np.float32),
                             team=team, al=al, rnd=rnd,
                             move=mcts.move_number(team, al, rnd)))
            act = mcts.search_action(env.board, team, al, rnd, evp, cfg_play, rng=rng)
            env.step(act)
        for r in recs:
            r["z"] = 1.0 if env.winner == r["team"] else -1.0
        with open(out, "ab") as f:
            pickle.dump(dict(seed=seed, game=ng, recs=recs, winner=env.winner), f)
        ng += 1
        npos += len(recs)
        print(f"[{seed}] game {ng} len {len(recs)} pos {npos} {time.time() - t0:.0f}s "
              f"({(time.time() - t0) / npos:.2f}s/pos)", flush=True)


if __name__ == "__main__":
    main()
