"""Scratch: raw-value calibration in low-material positions (each side K + <=1 piece), all 20 az_res_v20 live games.
Split: first 6 games (already in az_res_v6.md) vs games 7-20 (new). Models: v20, hand_v4f, az_res4f_v1, distill_v5hf."""
import sys
import numpy as np
sys.path.insert(0, "levels/level4"); sys.path.insert(0, "levels/level4/reviews")
from c1 import evaluator, MODELS
from analyze_matches import parse
from sg_pos import env_at
from v2_common import load
ms = load()
FIRST6 = {m["match_id"] for m in ms[:6]}
names = [n for n in (sys.argv[1:] or ["az_res_v20", "az_res4f_v2", "az_res4f_v3", "hand_v4f"]) if (MODELS / f"{n}.onnx").exists()]
evs = {n: evaluator(str(MODELS / f"{n}.onnx")) for n in names}
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
            rows.append((m["match_id"], k, env.current_team[:3], m["match_id"] in FIRST6, z, *[float(e(o)[1][0]) for e in evs.values()]))
print("low-mat positions", len(rows), "games", len({r[0] for r in rows}), sorted({r[0] for r in rows}))
side = np.array([r[2] for r in rows]); old = np.array([r[3] for r in rows]); R = np.array([r[4:] for r in rows], float); z = R[:, 0]
for subset, sel0 in [("all 20", np.ones(len(rows), bool)), ("games 7-20", ~old)]:
    print(f"--- {subset}")
    for s in ("CYA", "MAG"):
        for outcome, zz in (("mover LOST", -1), ("mover WON", 1)):
            sel = sel0 & (side == s) & (z == zz)
            if sel.any():
                print(f"  {s} to move, {outcome:10s} n={sel.sum():3d} games={len({rows[i][0] for i in np.where(sel)[0]})}: " + "  ".join(f"{n} {R[sel, 1+i].mean():+.2f}" for i, n in enumerate(names)))
    for i, n in enumerate(names):
        v = R[sel0, 1 + i]; zz = z[sel0]
        print(f"  {n:13s} MSE {np.mean((v - zz)**2):.3f} sign-acc {np.mean(np.sign(v) == zz):.2f}")
print("--- per game: first position with each side K + <=1 piece")
from collections import Counter
tally = Counter()
for m in ms:
    ks = [r for r in rows if r[0] == m["match_id"]]
    win = "CYAN" if "CYAN WINS" in m["protocol"] else "MAGENTA"; won = win == m["you"]
    if ks:
        k0 = ks[0][1]; mv, _ = parse(m["protocol"]); env = env_at(mv, k0)
        b = env.render().splitlines()[:8]; t = [x for l in b for x in l.split()[1:] if x[0] in "cm"]
        us = m["you"][0].lower(); ours = sorted(x for x in t if x[0] == us); theirs = sorted(x for x in t if x[0] != us)
        print(f"  {m['match_id']} {m['you'][:3]} {'WON ' if won else 'LOST'} lowmat from a{k0+1} r{env.round_number} us {ours} them {theirs}")
        tally[(m["you"][:3], "lowmat", won)] += 1
    else:
        tally[(m["you"][:3], "no-lowmat", won)] += 1
print(tally)
