"""Checks for the GPU self-play:
1. torch model (ckpt + scale) vs the deployed ONNX on real positions (max |diff| logits / value).
2. search_gen (generator) == referee mcts.run: same evaluator + same rng seed -> identical root visit counts.
3. (optional) replay a gsp_*.pkl through the real env: every stored action legal, outcomes match z.
uv run python levels/level4/agents/gpu/check.py [data/gsp_x.pkl]
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_sp as G  # noqa: E402
from c1 import MODELS, ColonyOneEnv, encoding, evaluator, mcts as M, CYAN  # noqa: E402


def drive(gen, ev):
    try:
        req = next(gen)
        while True:
            req = gen.send(ev(req))
    except StopIteration as e:
        return e.value


def positions(n=40, seed=0):
    rng = np.random.default_rng(seed)
    env, xs, states = ColonyOneEnv(), [], []
    while len(xs) < n:
        if env.terminated:
            env = ColonyOneEnv()
        xs.append(env.observation().astype(np.float32))
        idx = env.legal_action_indices()
        env.step(encoding.decode_action(int(rng.choice(idx)), env.current_team, env.board))
    return np.stack(xs)


def main():
    onnx_ev = evaluator(str(MODELS / "deriv_4f2_rsv125.onnx"))
    tev = G.TorchEval(G.load_torch_model("rsv125"))
    X = positions(200)
    l1, v1 = onnx_ev(X)
    l2, v2 = tev(X)
    print(f"[1] torch vs onnx rsv125: max|dlogit| {np.abs(l1 - l2).max():.2e}, max|dv| {np.abs(v1 - v2).max():.2e}")

    # [2] generator search == referee run (ONNX evaluator, same seed)
    env = ColonyOneEnv()
    cfg = M.MCTSConfig(simulations=100)
    rng0 = np.random.default_rng(1)
    same = 0
    for k in range(12):
        t, al, rn = env.current_team, env.actions_left, env.round_number
        r1 = M.run(env.board, t, al, rn, onnx_ev, cfg, rng=np.random.default_rng(100 + k))
        r2 = drive(G.search_gen(env.board, t, al, rn, cfg, np.random.default_rng(100 + k)), onnx_ev)
        same += M.visit_distribution(r1) == M.visit_distribution(r2)
        a = M.select_action_index(r1, 0.0, rng=rng0)
        env.step(encoding.decode_action(a, t, env.board))
    print(f"[2] generator search == referee mcts.run on {same}/12 positions")

    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        data = pickle.load(open(p, "rb"))
        meta = pickle.load(open(p.with_suffix(".meta.pkl"), "rb"))
        bad, off, zok = 0, 0, 0
        for acts, cw, n in zip(meta["actions"], meta["cyan_won"], meta["n"]):
            env = ColonyOneEnv()
            for i, a in enumerate(acts):
                rec = data[off + i]
                x = env.observation()
                ok = (np.array_equal(rec[0], x[:13].astype(np.uint8)) and a in set(env.legal_action_indices())
                      and int(a) in set(rec[3].tolist()))
                bad += not ok
                env.step(encoding.decode_action(a, env.current_team, env.board))
            zok += env.terminated and ((env.winner == CYAN) == cw)
            off += n
        assert off == len(data)
        print(f"[3] replay {len(meta['actions'])} games / {len(data)} records: bad records {bad}, "
              f"terminal+winner match {zok}/{len(meta['actions'])}")


if __name__ == "__main__":
    main()
