"""How do games between two models end? (firewall burn vs capture vs bare-kernel), per winner colour.

uv run python levels/level4/endings.py A.onnx B.onnx [games] [workers]
"""

import sys
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1 import CYAN, ColonyOneEnv, make_agent  # noqa: E402


def one(args):
    a, b, g = args
    a_cyan = g % 2 == 0
    A, B = make_agent(a, 100, 7 * g), make_agent(b, 100, 7 * g + 1)
    cy, mg = (A, B) if a_cyan else (B, A)
    env = ColonyOneEnv()
    log = []
    while not env.terminated:
        ag = cy if env.current_team == CYAN else mg
        act = ag.select(env.board, env.current_team, env.actions_left, env.round_number)
        log.append(env.format_action(act))
        rnd = env.round_number
        env.step(act)
    kernels = [p for p in env.board.pieces() if p.piece_type.name == "KERNEL"] \
        if hasattr(env.board, "pieces") else None
    last = log[-1]
    why = "capture" if "x" in last.lower() and "->" in last else "other"
    if why != "capture" and rnd in (8, 16, 24, 26) and env.action % 4 == 0:
        why = f"firewall r{rnd}"
    return ("A" if (env.winner == CYAN) == a_cyan else "B", env.winner, why, env.action, kernels)


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    w = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    with get_context("spawn").Pool(w) as pool:
        res = pool.map(one, [(a, b, g) for g in range(n)])
    c = Counter((r[0], r[1], r[2]) for r in res)
    for k, v in c.most_common():
        print(v, "winner", k[0], "colour", k[1], "end:", k[2])
    print("lengths", sorted(r[3] for r in res))
