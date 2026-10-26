"""Scratch: Kernel ring depth (0 = outer 8x8 ring, 3 = centre) for us vs them over the low-material phase of each v20 game."""
import sys
import numpy as np
sys.path.insert(0, "levels/level4"); sys.path.insert(0, "levels/level4/reviews")
from analyze_matches import parse
from sg_pos import env_at
from v20_common import load
for m in load():
    mv, _ = parse(m["protocol"]); win = "CYAN" if "CYAN WINS" in m["protocol"] else "MAGENTA"
    us = m["you"][0].lower(); d = []
    for k in range(len(mv)):
        env = env_at(mv, k); b = env.render().splitlines()[:8]
        grid = [l.split()[1:] for l in b]
        t = [x for l in grid for x in l if x[0] in "cm"]
        if sum(1 for x in t if x[0] == "c" and x != "cK") <= 1 and sum(1 for x in t if x[0] == "m" and x != "mK") <= 1:
            pos = {x: (r, c) for r, l in enumerate(grid) for c, x in enumerate(l) if x in ("cK", "mK")}
            ring = lambda p: min(p[0], 7 - p[0], p[1], 7 - p[1])
            d.append((ring(pos[f"{us}K"]), ring(pos[("m" if us == "c" else "c") + "K"])))
    if d:
        a = np.array(d)
        print(m["match_id"], m["you"][:3], "WON " if win == m["you"] else "LOST", f"lowmat positions {len(a):3d}  mean ring us {a[:,0].mean():.2f} them {a[:,1].mean():.2f}  share(us deeper) {np.mean(a[:,0]>a[:,1]):.2f} share(us shallower) {np.mean(a[:,0]<a[:,1]):.2f}  last us {a[-1,0]} them {a[-1,1]}")
