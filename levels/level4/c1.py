"""Shared Level 4 toolkit (COLONY: ONE tournament).

    import sys; sys.path.insert(0, "levels/level4"); from c1 import *

- ENV / colony_one imports with the Level 4 scene already installed as scene.json
- export_onnx(torch_model, path): export + Constant->initializer fix + official validator
- validate(path) -> (ok, text)
- arena(a, b, games, sims=100, workers=4): a/b are .onnx paths or "greedy"/"random".
  Sides alternate every game. Call it under `if __name__ == "__main__":` (Windows spawn).

CLI:
    uv run python levels/level4/c1.py validate model.onnx
    uv run python levels/level4/c1.py arena a.onnx b.onnx [games] [workers] [sims]
"""

from __future__ import annotations

import math
import secrets
import subprocess
import sys
import time
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ENV = HERE / "data" / "colony-one-env"
MODELS = HERE / "models"
if str(ENV) not in sys.path:
    sys.path.insert(0, str(ENV))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from colony_one import constants, encoding, mcts  # noqa: E402
from colony_one.baselines import GreedyAgent, RandomAgent  # noqa: E402
from colony_one.env import ColonyOneEnv  # noqa: E402
from colony_one.onnx_runner import OnnxEvaluator  # noqa: E402

CYAN, MAGENTA = constants.CYAN, constants.MAGENTA
TOURNAMENT_SIMS = 100


# ----------------------------------------------------------------------------- export

def fix_constants(path: Path) -> None:
    """The validator rejects the Constant op: turn every Constant node into an initializer."""
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(path))
    g = m.graph
    keep = []
    for node in g.node:
        if node.op_type != "Constant":
            keep.append(node)
            continue
        a = node.attribute[0]
        if a.name == "value":
            t = numpy_helper.to_array(a.t)
        elif a.name == "value_float":
            t = np.array(a.f, np.float32)
        elif a.name == "value_floats":
            t = np.array(list(a.floats), np.float32)
        elif a.name == "value_int":
            t = np.array(a.i, np.int64)
        elif a.name == "value_ints":
            t = np.array(list(a.ints), np.int64)
        else:
            raise ValueError(f"unsupported Constant attribute {a.name}")
        g.initializer.append(numpy_helper.from_array(t, node.output[0]))
    del g.node[:]
    g.node.extend(keep)
    # Exporters may leave symbolic dims (e.g. 'Reshapepolicy_dim_1'); the validator wants
    # observation [batch,16,8,8], policy [batch,4096], value [batch] with one batch name.
    want = {g.input[0].name: ["batch", 16, 8, 8], g.output[0].name: ["batch", 4096],
            g.output[1].name: ["batch"]}
    for vi in list(g.input[:1]) + list(g.output[:2]):
        dims = vi.type.tensor_type.shape.dim
        del dims[:]
        for d in want[vi.name]:
            nd = dims.add()
            if isinstance(d, str):
                nd.dim_param = d
            else:
                nd.dim_value = d
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def validate(path) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, str(ENV / "tools" / "validate_submission.py"), str(path)],
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    return "ACCEPTED" in r.stdout, out[-2500:]


def export_onnx(model, path, check: bool = True) -> Path:
    """Export a torch module mapping (B,16,8,8) -> (policy (B,4096), value (B,)).

    Rules that keep the graph validator-clean:
      * reshape with literal ints only: x.flatten(1), x.reshape(-1, 4096), v.reshape(-1)
        (never x.view(x.size(0), ...) -- that makes a data-dependent shape)
      * value must be (B,) in [-1, 1] (end with tanh), policy raw logits
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu().float()
    x = torch.zeros(3, 16, 8, 8)
    with torch.no_grad():
        torch.onnx.export(
            model, (x,), str(path), input_names=["observation"],
            output_names=["policy", "value"], opset_version=17,
            dynamic_axes={"observation": {0: "batch"}, "policy": {0: "batch"},
                          "value": {0: "batch"}},
            do_constant_folding=True, dynamo=False, external_data=False)
    fix_constants(path)
    if check:
        ok, msg = validate(path)
        if not ok:
            raise RuntimeError(f"validator rejected {path}:\n{msg}")
    return path


# ----------------------------------------------------------------------------- agents

@lru_cache(maxsize=4)  # bounded: long-lived workers would otherwise keep every generation's session
def evaluator(path: str) -> OnnxEvaluator:
    return OnnxEvaluator(Path(path))


class SearchAgent:
    """Referee MCTS driven by an evaluator callable (B,16,8,8) -> ((B,4096),(B,))."""

    def __init__(self, ev, sims: int = TOURNAMENT_SIMS, seed: int | None = None):
        self.ev = ev
        self.cfg = mcts.MCTSConfig(simulations=sims)
        self.rng = np.random.default_rng(seed)

    def select(self, board, team, actions_left, round_number):
        return mcts.search_action(board, team, actions_left, round_number,
                                  self.ev, self.cfg, rng=self.rng)


def make_agent(spec: str, sims: int, seed: int):
    if spec == "greedy":
        return GreedyAgent(seed=seed)
    if spec == "random":
        return RandomAgent(seed=seed)
    return SearchAgent(evaluator(str(Path(spec).resolve())), sims, seed)


def play_game(cyan_agent, magenta_agent) -> tuple[str, int]:
    env = ColonyOneEnv()
    while not env.terminated:
        agent = cyan_agent if env.current_team == CYAN else magenta_agent
        env.step(agent.select(env.board, env.current_team, env.actions_left, env.round_number))
    return env.winner, env.action


def _arena_game(args):
    a, b, g, sims, seed = args
    a_cyan = g % 2 == 0
    A, B = make_agent(a, sims, seed + 2 * g), make_agent(b, sims, seed + 2 * g + 1)
    t = time.process_time()
    winner, n = play_game(A, B) if a_cyan else play_game(B, A)
    return (winner == CYAN) == a_cyan, a_cyan, n, time.process_time() - t


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def arena(a, b, games: int = 40, sims: int = TOURNAMENT_SIMS, workers: int = 4,
          seed: int | None = None, verbose: bool = True) -> dict:
    """Win rate of `a` vs `b` (sides alternate). Must run under __main__ guard on Windows."""
    seed = secrets.randbits(30) if seed is None else seed
    t0 = time.time()
    jobs = [(str(a), str(b), g, sims, seed) for g in range(games)]
    with get_context("spawn").Pool(workers) as pool:
        res = pool.map(_arena_game, jobs, chunksize=1)
    wins = sum(r[0] for r in res)
    cyan = [r for r in res if r[1]]
    mag = [r for r in res if not r[1]]
    lo, hi = wilson(wins, games)
    p = min(max(wins / games, 1e-3), 1 - 1e-3)
    out = dict(a=Path(str(a)).name, b=Path(str(b)).name, games=games, wins=wins,
               rate=wins / games, ci=(lo, hi), elo=400 * math.log10(p / (1 - p)),
               cyan_rate=sum(r[0] for r in cyan) / max(1, len(cyan)),
               magenta_rate=sum(r[0] for r in mag) / max(1, len(mag)),
               avg_actions=float(np.mean([r[2] for r in res])),
               cpu_per_game=float(np.mean([r[3] for r in res])),
               secs=time.time() - t0)
    if verbose:
        print(f"{out['a']} vs {out['b']}: {wins}/{games} = {out['rate']:.0%} "
              f"[{lo:.0%},{hi:.0%}] elo {out['elo']:+.0f} | as cyan {out['cyan_rate']:.0%} "
              f"as magenta {out['magenta_rate']:.0%} | {out['avg_actions']:.0f} actions, "
              f"{out['cpu_per_game']:.1f}s cpu/game (both sides), {out['secs']:.0f}s",
              flush=True)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "validate":
        ok, msg = validate(sys.argv[2])
        print(msg)
    elif cmd == "arena":
        args = sys.argv[2:]
        arena(args[0], args[1], games=int(args[2]) if len(args) > 2 else 40,
              workers=int(args[3]) if len(args) > 3 else 4,
              sims=int(args[4]) if len(args) > 4 else TOURNAMENT_SIMS)
