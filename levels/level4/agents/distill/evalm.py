"""uv run python levels/level4/agents/distill/evalm.py MODEL.onnx GAMES WORKERS opp1 [opp2 ...]  (opp = path, greedy, random)"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from c1 import arena  # noqa: E402

if __name__ == "__main__":
    model, games, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    for opp in sys.argv[4:]:
        r = arena(model, opp, games=games, workers=workers)
        r["time"] = time.strftime("%H:%M:%S", time.gmtime())
        with open(HERE / "arena_results.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
