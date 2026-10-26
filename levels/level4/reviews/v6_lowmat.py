"""Scratch: raw-value calibration in low-material positions (K+<=1 vs K+<=1) across v6 + v20 live games."""
import json, sys
import numpy as np
sys.path.insert(0, "levels/level4"); sys.path.insert(0, "levels/level4/reviews")
from c1 import evaluator, MODELS
from analyze_matches import parse
from sg_pos import env_at
ms = [m for f, lo, hi in [("matches_az_res_v6.json", "2026-10-23T14:43:30", "2026-10-23T14:53:30"), ("matches_az_res_v20_a.json", "2026-10-23T14:53:30", "2026-10-23T23:00")]
      for m in json.load(open(f"levels/level4/data/matches/{f}")) if lo <= m["played_at"] <= hi and m.get("protocol")]
print("games", len(ms))
evs = {n: evaluator(str(MODELS / f"{n}.onnx")) for n in ["az_res_v6", "az_res_v20", "hand_v4f"]}
rows = []
for m in ms:
    mv, _ = parse(m["protocol"]); win = "CYAN" if "CYAN WINS" in m["protocol"] else "MAGENTA"
    for k in range(len(mv)):
        env = env_at(mv, k)
        b = env.render().splitlines()[:8]; t = [x for l in b for x in l.split()[1:] if x[0] in "cm"]
        nc = sum(1 for x in t if x[0] == "c" and x != "cK"); nm = sum(1 for x in t if x[0] == "m" and x != "mK")
        if nc <= 1 and nm <= 1:
            z = 1.0 if env.current_team == win else -1.0
            o = env.observation()[None]
            rows.append((m["match_id"], k, env.current_team[:3], nc, nm, z, *[float(e(o)[1][0]) for e in evs.values()]))
R = np.array([r[5:] for r in rows])
print("low-mat positions", len(rows), "from games", sorted({r[0] for r in rows}))
for i, n in enumerate(evs):
    v = R[:, 1 + i]; z = R[:, 0]
    print(f"{n:11s} MSE vs outcome {np.mean((v - z)**2):.3f}  mean(v*z) {np.mean(v*z):+.3f}  sign acc {np.mean(np.sign(v) == z):.2f}  mean v when mover LOSES {v[z<0].mean():+.2f} (n={np.sum(z<0)})  when mover WINS {v[z>0].mean():+.2f}")
# per side: magenta-to-move positions in MAG-lost games
for side in ("CYA", "MAG"):
    sel = np.array([r[2] == side for r in rows])
    if sel.any():
        print(side, "to move:", " ".join(f"{n}: v|lose {R[sel & (R[:,0]<0), 1+i].mean():+.2f}" for i, n in enumerate(evs)), "n lose", int(np.sum(sel & (R[:,0]<0))), "n win", int(np.sum(sel & (R[:,0]>0))))
