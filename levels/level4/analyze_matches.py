"""Summarise downloaded tournament matches: how each game ended and why.

Run with PyCharm ▶ or:  uv run python levels/level4/analyze_matches.py [matches.json]
Also replays every protocol through the engine (checks we can turn replays into training data).
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1 import CYAN, ColonyOneEnv  # noqa: E402

HERE = Path(__file__).parent
DEFAULT = HERE / "data" / "matches" / "tournament-matches.json"
LINE = re.compile(r"^\s*(\d+)\.\s+(CYAN|MAGENTA)\s+(\w\w)\s+(\w\d)\s+->\s+(\w\d)\s+(MOVE|SHOOT)")


def parse(protocol: str):
    moves, events = [], []
    for ln in protocol.splitlines():
        m = LINE.match(ln)
        if m:
            moves.append(m.groups()[1:])
        elif ln.startswith("----"):
            events.append(ln.strip("- *"))
    return moves, events


def replay(moves):
    """Replay protocol moves through the engine; returns env (raises on mismatch)."""
    env = ColonyOneEnv()
    for team, _piece, a, b, _kind in moves:
        assert env.current_team == team, (env.current_team, team)
        cands = [x for x in env.legal_actions() if env.format_action(x).replace(" ", "")
                 .upper().startswith(f"{a}->{b}".upper()) or
                 (a in env.format_action(x) and b in env.format_action(x))]
        if len(cands) != 1:
            raise ValueError(f"cannot match {a}->{b}: {[env.format_action(x) for x in cands]}")
        env.step(cands[0])
    return env


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    matches = json.loads(path.read_text())
    reasons, rows = Counter(), []
    for m in matches:
        moves, events = parse(m["protocol"])
        winner = "CYAN" if "CYAN WINS" in m["protocol"] else "MAGENTA"
        won = winner == m["you"]
        fw = [e for e in events if "FIREWALL" in e]
        last = moves[-1] if moves else None
        if any("BARE KERNEL" in e for e in events):
            why = "bare-kernel rule"
        elif fw and ("KE" in fw[-1]):
            why = "firewall killed kernel(s): " + ",".join(
                x for x in re.findall(r"(CYAN|MAGENTA)xKE", fw[-1]))
        elif last and "xKE" in m["protocol"].splitlines()[-3]:
            why = f"kernel captured by {last[0]} {last[1]}"
        else:
            why = "other: " + (events[-2] if len(events) > 1 else "?")
        reasons[(m["you"], "WON" if won else "LOST", why.split(":")[0])] += 1
        rows.append((m["match_id"], m["you"], "WON " if won else "LOST", len(moves), m.get("opponent") or "?", why))
        try:
            replay(moves)
        except Exception as e:  # noqa: BLE001
            rows[-1] = rows[-1] + (f"REPLAY FAIL {e}",)
    for r in sorted(rows):
        print(*r)
    print()
    for k, v in reasons.most_common():
        print(v, k)


if __name__ == "__main__":
    main()
