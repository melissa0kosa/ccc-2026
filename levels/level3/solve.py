"""Level 3: build model.onnx (material-value heuristic), validate it, test vs GreedyAgent.

Run with PyCharm ▶ or:  uv run python levels/level3/solve.py [games]
Writes levels/level3/out/model.onnx  -> upload that file.
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).parent
ENV = HERE / "data" / "colony-one-env"
OUT = HERE / "out"
sys.path.insert(0, str(ENV))

from colony_one import constants, mcts  # noqa: E402
from colony_one.baselines import GreedyAgent  # noqa: E402
from colony_one.env import ColonyOneEnv  # noqa: E402
from colony_one.onnx_runner import OnnxEvaluator  # noqa: E402

# planes 0-5 own / 6-11 enemy: Ping, Daemon, Exploit, Trace, Ice, Kernel
PIECE_VALUE = [15.0, 45.0, 25.0, 30.0, 50.0, 100.0]
SCALE = 1 / 40


class MaterialNet(nn.Module):
    def __init__(self):
        super().__init__()
        # per-plane weight repeated over the 64 squares; planes 12-15 get 0.
        # Only Flatten/MatMul/Add/Tanh so the graph has no Constant nodes.
        w = torch.tensor(PIECE_VALUE + [-v for v in PIECE_VALUE] + [0.0] * 4) * SCALE
        self.register_buffer("w", w.repeat_interleave(64))       # (1024,)
        self.register_buffer("zero", torch.zeros(1024, 1))
        self.register_buffer("prior", torch.zeros(1, 4096))

    def forward(self, x):
        flat = torch.flatten(x, 1)                               # (B, 1024)
        value = torch.tanh(flat @ self.w)                        # (B,)
        policy = flat @ self.zero + self.prior                   # (B, 4096)
        return policy, value


def export(model: nn.Module, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    x = torch.zeros(2, 16, 8, 8)
    torch.onnx.export(model, (x,), str(path), input_names=["observation"],
                      output_names=["policy", "value"], opset_version=17,
                      dynamic_axes={"observation": {0: "batch"}, "policy": {0: "batch"},
                                    "value": {0: "batch"}},
                      dynamo=False, external_data=False)
    return path


def validate(path: Path) -> bool:
    r = subprocess.run([sys.executable, str(ENV / "tools" / "validate_submission.py"), str(path)],
                       capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:])
    return "ACCEPTED" in r.stdout


def play_vs_greedy(path: Path, games: int, sims: int = 60, as_cyan: bool = True) -> float:
    ev = OnnxEvaluator(path)
    cfg = mcts.MCTSConfig(simulations=sims)
    me = constants.CYAN if as_cyan else constants.MAGENTA
    wins, t0 = 0, time.time()
    for g in range(games):
        rng = np.random.default_rng(1000 + g)
        bot = GreedyAgent(seed=g)
        env = ColonyOneEnv()
        while not env.terminated:
            if env.current_team == me:
                a = mcts.search_action(env.board, env.current_team, env.actions_left,
                                       env.round_number, ev, cfg, rng=rng)
            else:
                a = bot.select(env.board, env.current_team, env.actions_left, env.round_number)
            env.step(a)
        wins += env.winner == me
    rate = wins / games
    print(f"vs Greedy as {me}: {wins}/{games} = {rate:.0%}  ({(time.time() - t0) / games:.1f}s/game)")
    return rate


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    path = export(MaterialNet(), OUT / "model.onnx")
    print("exported", path, path.stat().st_size, "bytes")
    if validate(path):
        play_vs_greedy(path, games)


if __name__ == "__main__":
    main()
