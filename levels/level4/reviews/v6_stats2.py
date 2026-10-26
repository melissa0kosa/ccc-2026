"""Scratch: disagreement stats + near-disasters from v6_scan.json."""
import json, sys
import numpy as np
sys.path.insert(0, "levels/level4"); sys.path.insert(0, "levels/level4/reviews")
from v6_common import load
from sg_stats import classify
R = json.load(open("levels/level4/reviews/v6_scan.json"))
won = {m["match_id"]: classify(m)[1] for m in load()}
def tab(rows, name):
    d = [r for r in rows if r["c100"] != r["c800"]]
    dq = np.array([r["q800_c800"] - r["q800_c100"] for r in d]) if d else np.zeros(0)
    allg = np.array([r["q800_c800"] - r["q800_c100"] for r in rows])
    print(f"{name}: n={len(rows)} diff={len(d)} ({len(d)/len(rows):.1%}) median dQ={np.median(dq) if len(dq) else 0:.3f} >0.1={np.sum(dq>0.1)} >0.2={np.sum(dq>0.2)} >0.3={np.sum(dq>0.3)} meangain={allg.mean():.3f}")
tab(R, "all"); tab([r for r in R if r["k"] >= 8], "post-warmup"); tab([r for r in R if r["k"] < 8], "warmup")
w = [r for r in R if r["k"] < 8]
ql = [r["q800_c800"] - r["q800_played"] for r in w]; ql2 = [r["q800_c800"] - r["q800_played"] for r in R if r["k"] >= 8]
print("warmup: played!=c100", sum(r["played"] != r["c100"] for r in w), "/", len(w), " mean Qloss(played) warm", np.nanmean(ql), "post", np.nanmean(ql2))
print("warmup Qloss>0.2:", [(r["mid"], f"a{r['k']+1}", r["played"], round(r["p_played"], 2), round(r["q800_c800"] - r["q800_played"], 2)) for r in w if r["q800_c800"] - r["q800_played"] > 0.2])
print("post-warmup played != local c100:", [(r["mid"], r["k"] + 1, r["played"], r["c100"]) for r in R if r["k"] >= 8 and r["played"] != r["c100"]])
print("top dQ:", sorted([(round(r["q800_c800"] - r["q800_c100"], 2), r["mid"], f"a{r['k']+1}", r["c100"], r["c800"], round(r["p_c800"], 2)) for r in R if r["c100"] != r["c800"]], reverse=True)[:12])
for mid in sorted(won):
    rr = [r for r in R if r["mid"] == mid]
    mn = min(rr, key=lambda r: r["root800"])
    print(mid, "WON" if won[mid] else "LOST", "min root800", round(mn["root800"], 2), f"a{mn['k']+1}", " v-root800 corr", round(np.corrcoef([r["v"] for r in rr], [r["root800"] for r in rr])[0, 1], 2))
v = np.array([r["v"] for r in R]); ro = np.array([r["root800"] for r in R])
print("raw v vs root800: corr", np.corrcoef(v, ro)[0, 1].round(3), "MAE", np.abs(v - ro).mean().round(3), "|v|>0.9", np.mean(np.abs(v) > 0.9).round(3))
pt = np.array([r["p_top"] for r in R]); print("prior top1 pct 10/25/50/75/90", np.percentile(pt, [10, 25, 50, 75, 90]).round(3), "c800==prior top", np.mean([r["c800"] == r["top"] for r in R]).round(2), "median p(c800)", np.median([r["p_c800"] for r in R]).round(3))
print("big |v-root800|>0.5:", [(r["mid"], f"a{r['k']+1}", round(r["v"], 2), round(r["root800"], 2)) for r in R if abs(r["v"] - r["root800"]) > 0.5])
