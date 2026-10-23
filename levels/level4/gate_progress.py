"""Tally the gatekeeper's live game logs (levels/level4/gate/*.jsonl): wins per pairing.

uv run python levels/level4/gate_progress.py [N newest files]
"""

import collections
import json
import sys
from pathlib import Path

GATE = Path(__file__).parent / "gate"


def name(x) -> str:
    return Path(str(x).replace("\\", "/")).name


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    files = sorted(GATE.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:n]
    for f in files:
        rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        keys = sorted(rows[0].keys())
        tally = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            a = r.get("a") or r.get("cand") or r.get("model_a") or r.get("A")
            b = r.get("b") or r.get("ref") or r.get("model_b") or r.get("B")
            won = r.get("a_won", r.get("win", r.get("won", r.get("a_win"))))
            t = tally[(name(a), name(b))]
            t[0] += 1
            t[1] += bool(won)
        print(f"== {f.name}  (keys: {keys[:8]})")
        for (a, b), (games, wins) in tally.items():
            print(f"   {a} vs {b}: {wins}/{games} = {wins / games:.0%}")


if __name__ == "__main__":
    main()
