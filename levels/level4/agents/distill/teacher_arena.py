"""Teacher (alpha-beta, plays its best move directly) vs an ONNX model under the referee search.
uv run python levels/level4/agents/distill/teacher_arena.py OPP.onnx GAMES"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from teacher import Searcher, CYAN  # noqa: E402
from c1 import make_agent, play_game, wilson  # noqa: E402


class TeacherAgent:
    def __init__(self, seed):
        self.S = Searcher()
        self.rng = np.random.default_rng(seed)

    def select(self, board, team, al, rnd):
        acts, sc, best = self.S.root(board, team, al, rnd)
        sc = np.array(sc) + self.rng.random(len(sc)) * 1e-3
        return acts[int(np.argmax(sc))]


if __name__ == "__main__":
    opp, games = sys.argv[1], int(sys.argv[2])
    wins, cy = 0, [0, 0]
    t0 = time.time()
    for g in range(games):
        T = TeacherAgent(g)
        O = make_agent(str(Path(opp).resolve()), 100, 1000 + g)
        tc = g % 2 == 0
        w, n = play_game(T, O) if tc else play_game(O, T)
        won = (w == CYAN) == tc
        wins += won
        if tc:
            cy[0] += won
        else:
            cy[1] += won
        print(f"game {g} teacher {'cyan' if tc else 'magenta'} {'W' if won else 'L'} {n} actions", flush=True)
    lo, hi = wilson(wins, games)
    print(f"teacher vs {Path(opp).name}: {wins}/{games} = {wins/games:.0%} [{lo:.0%},{hi:.0%}] "
          f"cyan {cy[0]}/{(games+1)//2} magenta {cy[1]}/{games//2} {time.time()-t0:.0f}s", flush=True)
