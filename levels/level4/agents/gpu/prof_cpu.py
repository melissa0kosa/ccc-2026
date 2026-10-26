"""Profile one referee-MCTS self-play game on CPU (ONNX evaluator) to see engine vs NN split."""
import cProfile, pstats, sys, time
from pathlib import Path
L4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(L4))
import numpy as np
from c1 import evaluator, ColonyOneEnv, mcts, encoding, MODELS

ev0 = evaluator(str(MODELS / "deriv_4f2_rsv125.onnx"))
tnn = [0.0, 0, 0]
def ev(x):
    t = time.perf_counter(); r = ev0(x); tnn[0] += time.perf_counter() - t; tnn[1] += 1; tnn[2] += len(x); return r

def game(seed):
    rng = np.random.default_rng(seed)
    cfg = mcts.MCTSConfig(simulations=100)
    env = ColonyOneEnv(); n = 0
    while not env.terminated:
        root = mcts.run(env.board, env.current_team, env.actions_left, env.round_number, ev, cfg, rng=rng)
        temp = mcts.warmup_temperature(mcts.move_number(env.current_team, env.actions_left, env.round_number))
        a = mcts.select_action_index(root, temp, rng=rng)
        env.step(encoding.decode_action(a, env.current_team, env.board)); n += 1
    return n
t = time.perf_counter(); ns = [game(s) for s in range(3)]; dt = time.perf_counter() - t
print(f"3 games {dt:.2f}s, actions {ns}, nn time {tnn[0]:.2f}s calls {tnn[1]} positions {tnn[2]}")
pr = cProfile.Profile(); pr.enable(); game(7); pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(14)
