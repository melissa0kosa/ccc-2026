"""Scratch: live stats for hand_v2 (matches >= 14:01:30)."""
import json, re, sys
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "reviews"))
from analyze_matches import parse
from sg_stats import classify

def load(lo="2026-10-23T14:01:30"):
    ms = json.loads((HERE / "data/matches/matches_hand_v2_1.json").read_text())
    return [m for m in ms if m["played_at"] >= lo and m.get("protocol")]

if __name__ == "__main__":
    c = Counter()
    for m in sorted(load(), key=lambda m: m["played_at"]):
        w, won, n, why, ev = classify(m)
        c[(m["you"], won)] += 1
        moves, _ = parse(m["protocol"])
        us = m["you"]
        caps_us = [f"a{i+1}:{mv[1]}x" + l.split("x")[-1][:2] for i, (mv, l) in enumerate(zip(moves, [l for l in m["protocol"].splitlines() if re.match(r"^\s*\d+\.", l)])) if mv[0] == us and "x" in l.split("->")[-1]]
        caps_them = [f"a{i+1}:{mv[1]}x" + l.split("x")[-1][:2] for i, (mv, l) in enumerate(zip(moves, [l for l in m["protocol"].splitlines() if re.match(r"^\s*\d+\.", l)])) if mv[0] != us and "x" in l.split("->")[-1]]
        fw = [e for e in ev if "FIREWALL" in e or "BARE" in e]
        print(m["match_id"], m["played_at"][11:19], us[:3], "WON " if won else "LOST", n, m.get("opponent") or "?", "score", m.get("score"), "|", why)
        print("    our caps:", caps_us, " their caps:", caps_them)
        print("    events:", fw)
    print(c)
