"""Arena a candidate vs opponents (100 sims, colour split), append to arena_results.jsonl.

    uv run python levels/level4/agents/prior/run_arena.py <cand.onnx> <games> <opp1.onnx> [opp2.onnx ...]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from c1 import arena  # noqa: E402

if __name__ == "__main__":
    cand, games = sys.argv[1], int(sys.argv[2])
    for opp in sys.argv[3:]:
        r = arena(cand, opp, games=games, workers=3)
        with open(HERE / "arena_results.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
