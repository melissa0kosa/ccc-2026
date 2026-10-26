"""Scratch: live stats for stopgap_v1 (matches >= 13:52) vs seat_l3_material."""
import json, re, sys
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from analyze_matches import parse, replay

def load(fn, lo=None, hi=None):
    ms = json.loads((HERE / "data" / "matches" / fn).read_text())
    return [m for m in ms if (lo is None or m["played_at"] >= lo) and (hi is None or m["played_at"] < hi)]

def classify(m):
    moves, events = parse(m["protocol"])
    p = m["protocol"]
    winner = "CYAN" if "CYAN WINS" in p else "MAGENTA"
    won = winner == m["you"]
    fw = [e for e in events if "FIREWALL" in e]
    lines = [l for l in p.splitlines() if l.strip()]
    if any("BARE" in e for e in events):
        why = "bare-kernel"
    elif fw and "KE" in fw[-1]:
        why = "firewall " + fw[-1]
    elif "xKE" in "".join(lines[-3:]):
        last = moves[-1]
        why = f"kernel captured by {last[0]} {last[1]} {last[4]}"
    else:
        why = "other " + str(events[-2:])
    return winner, won, len(moves), why, events

if __name__ == "__main__":
    for name, ms in [("stopgap_v1", load("matches_1405_stopgap_v1.json", "2026-10-23T13:51:00")),
                     ("prev(all in tournament-matches)", load("tournament-matches.json"))]:
        print("=====", name, len(ms))
        c = Counter()
        for m in sorted(ms, key=lambda m: m["played_at"]):
            w, won, n, why, ev = classify(m)
            c[(m["you"], won)] += 1
            print(m["match_id"], m["played_at"][11:19], m["you"][:3], "WON " if won else "LOST", n, m.get("opponent") or "?", m.get("score"), "|", why, "|", [e for e in ev if "FIREWALL" in e or "BARE" in e][:4])
        print(c)
