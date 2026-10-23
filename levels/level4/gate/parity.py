"""Output parity of two ONNX models on positions from real games (greedy/random/mcts mix).
usage: parity.py A.onnx B.onnx [games]"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from c1 import CYAN, ColonyOneEnv, MODELS, encoding, evaluator, GreedyAgent, RandomAgent  # noqa: E402

a, b = MODELS / sys.argv[1], MODELS / sys.argv[2]
games = int(sys.argv[3]) if len(sys.argv) > 3 else 60
ea, eb = evaluator(str(a.resolve())), evaluator(str(b.resolve()))
rng = np.random.default_rng(0)
X, M = [], []
for g in range(games):
    env = ColonyOneEnv()
    ag = [GreedyAgent(seed=g), RandomAgent(seed=g)] if g % 2 else [GreedyAgent(seed=g), GreedyAgent(seed=g + 1000)]
    while not env.terminated:
        X.append(encoding.encode_board(env.board, env.current_team, env.actions_left, env.round_number))
        M.append(env.legal_mask())
        agent = ag[0] if env.current_team == CYAN else ag[1]
        env.step(agent.select(env.board, env.current_team, env.actions_left, env.round_number))
X = np.stack(X).astype(np.float32)
M = np.stack(M).astype(bool)
pa, va = ea(X)
pb, vb = eb(X)
dv = np.abs(va - vb)
la = np.where(M, pa, -np.inf)
lb = np.where(M, pb, -np.inf)


def sm(l):
    l = l - l.max(1, keepdims=True)
    e = np.exp(l)
    return e / e.sum(1, keepdims=True)


sa, sb = sm(la), sm(lb)
dp = np.abs(sa - sb).max(1)
amax = (la.argmax(1) != lb.argmax(1)).mean()
print(f"{a.name} vs {b.name}: {len(X)} positions | value |dv| max {dv.max():.3g} mean {dv.mean():.3g} "
      f"(>0.05: {(dv > 0.05).mean():.1%}) | legal-softmax max|dp| max {dp.max():.3g} mean {dp.mean():.3g} "
      f"(>0.05: {(dp > 0.05).mean():.1%}) | prior argmax differs {amax:.1%}")
