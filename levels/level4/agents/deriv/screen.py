"""Sequential arena screens, logged with sha256 of both files.

uv run python levels/level4/agents/deriv/screen.py GAMES WORKERS opp.onnx cand1.onnx [cand2.onnx ...]
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
LOG = HERE / "arena_log.jsonl"


def sha(p):
    if p in ("greedy", "random"):
        return p
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if __name__ == "__main__":
    from c1 import arena
    games, workers, opp = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    for cand in sys.argv[4:]:
        r = arena(cand, opp, games=games, workers=workers)
        r["sha_a"], r["sha_b"] = sha(cand), sha(opp)
        r["utc"] = time.strftime("%H:%M:%S", time.gmtime())
        with open(LOG, "a") as f:
            f.write(json.dumps(r) + "\n")
