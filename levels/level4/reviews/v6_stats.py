"""Scratch: live stats for az_res_v6 (14:43:30-14:53:30)."""
import json, re, sys
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "reviews"))
from analyze_matches import parse
from sg_stats import classify
from v6_common import load

if __name__ == "__main__":
    allm = load(False)
    print("entries in window:", len(allm), "with protocol:", sum(1 for m in allm if m.get("protocol")))
    for m in allm:
        if not m.get("protocol"):
            print("NO PROTOCOL:", m["match_id"], m["played_at"], m["you"], m.get("opponent"), m.get("score"))
    c = Counter(); caps = Counter()
    for m in sorted(load(), key=lambda m: m["played_at"]):
        w, won, n, why, ev = classify(m)
        c[(m["you"], won)] += 1
        moves, _ = parse(m["protocol"])
        us = m["you"]
        L = [l for l in m["protocol"].splitlines() if re.match(r"^\s*\d+\.", l)]
        caps_us = [f"a{i+1}:{mv[1]}x" + l.split("x")[-1][:2] for i, (mv, l) in enumerate(zip(moves, L)) if mv[0] == us and "x" in l.split("->")[-1]]
        caps_them = [f"a{i+1}:{mv[1]}x" + l.split("x")[-1][:2] for i, (mv, l) in enumerate(zip(moves, L)) if mv[0] != us and "x" in l.split("->")[-1]]
        caps["us"] += len(caps_us); caps["them"] += len(caps_them)
        fw = [e for e in ev if "FIREWALL" in e or "BARE" in e]
        print(m["match_id"], m["played_at"][11:19], us[:3], "WON " if won else "LOST", n, m.get("opponent") or "?", "score", m.get("score"), "|", why)
        print("    our caps:", caps_us, " their caps:", caps_them)
        print("    events:", fw)
    print(c, caps)
