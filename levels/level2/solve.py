"""Level 2: find CYAN's forced win. Edit SCENE (and ROUND) to the board on screen, then
run with PyCharm ▶ or:  uv run python levels/level2/solve.py
Uppercase = CYAN, lowercase = MAGENTA.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import solve_scene  # noqa: E402

SCENE = {"A1": "K", "B4": "P", "H8": "k"}
ROUND = 1

if __name__ == "__main__":
    t = time.time()
    depth, move = solve_scene(SCENE, ROUND)
    print(f"PLAY: {move}   (forced win in {depth} CYAN turns, {time.time() - t:.1f}s)")
