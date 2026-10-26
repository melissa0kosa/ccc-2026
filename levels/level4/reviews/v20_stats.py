"""Scratch: live stats for az_res_v20 (>= 14:53:30)."""
import re, sys
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "reviews"))
from analyze_matches import parse
from sg_stats import classify
from v20_common import load

if __name__ == "__main__":
    allm = load(False)
    print("entries:", len(allm), "with protocol:", sum(1 for m in allm if m.get("protocol")), "keys", list(allm[0].keys()))
    for m in allm:
        if not m.get("protocol"):
            print("NO PROTOCOL:", m["match_id"], m["played_at"], m["you"], m.get("opponent"), m.get("score"))
    c = Counter()
    for i, m in enumerate(allm):
        if not m.get("protocol"):
            continue
        w, won, n, why, ev = classify(m)
        c[(m["you"][:3], won)] += 1
        moves, _ = parse(m["protocol"]); us = m["you"]
        L = [l for l in m["protocol"].splitlines() if re.match(r"^\s*\d+\.", l)]
        cu = [f"a{j+1}:{mv[1]}x" + l.split("x")[-1][:2] for j, (mv, l) in enumerate(zip(moves, L)) if mv[0] == us and "x" in l.split("->")[-1]]
        ct = [f"a{j+1}:{mv[1]}x" + l.split("x")[-1][:2] for j, (mv, l) in enumerate(zip(moves, L)) if mv[0] != us and "x" in l.split("->")[-1]]
        sc_ok = (m.get("score") == 1) == (w == "CYAN")
        print(f"g{i+1:2d}", m["match_id"], m["played_at"][11:19], us[:3], "WON " if won else "LOST", n, m.get("opponent") or "?", "score", m.get("score"), "ok" if sc_ok else "SCORE-MISMATCH", m.get("_proto_from", ""), "|", why)
        print("     our caps:", cu, " their caps:", ct, " ev:", [e for e in ev if "FIREWALL" in e or "BARE" in e])
    print(c)
