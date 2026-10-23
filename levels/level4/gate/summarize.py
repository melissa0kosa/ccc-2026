"""Summarize gate/games.jsonl per pairing (A vs B, pooled over runs).
Distinct games: cluster key (a_cyan, full action-sequence hash). n_eff = n^2 / sum(c_i^2).
Wilson CI on n (naive) and on n_eff (dedup-aware)."""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summarize(rows):
    n = len(rows)
    w = sum(r["a_won"] for r in rows)
    p = w / n
    cy = [r["a_won"] for r in rows if r["a_cyan"]]
    mg = [r["a_won"] for r in rows if not r["a_cyan"]]
    cl = Counter((r["a_cyan"], r["h"]) for r in rows)
    op = Counter((r["a_cyan"], r["h8"]) for r in rows)
    neff = n * n / sum(c * c for c in cl.values())
    hbrows = [r for r in rows if r.get("hb")]
    bcl = Counter((r["a_cyan"], r["hb"]) for r in hbrows)
    neff_b = (len(hbrows) ** 2 / sum(c * c for c in bcl.values()) * n / len(hbrows)) if hbrows else float("nan")
    # outcome consistency inside position clusters (same position after warmup -> same result?)
    incons = sum(1 for k in bcl if bcl[k] > 1 and len({r["a_won"] for r in hbrows if (r["a_cyan"], r["hb"]) == k}) > 1)
    lo, hi = wilson(w, n)
    elo_, ehi = wilson(p * neff, neff)
    return dict(n=n, w=w, p=p, lo=lo, hi=hi, cyan=sum(cy) / max(1, len(cy)), mag=sum(mg) / max(1, len(mg)),
                ncy=len(cy), nmg=len(mg), distinct=len(cl), openings=len(op), maxdup=max(cl.values()),
                neff=neff, nhb=len(hbrows), bdistinct=len(bcl), neff_b=neff_b, incons=incons, elo=elo_, ehi=ehi, cpu=sum(r["cpu"] for r in rows) / n,
                act=sum(r["n"] for r in rows) / n)


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "games.jsonl"
    by = defaultdict(list)
    for ln in path.read_text().splitlines():
        r = json.loads(ln)
        by[(r["a"], r["b"])].append(r)
    for (a, b), rows in by.items():
        s = summarize(rows)
        print(f"{a:18s} vs {b:16s} {s['w']:3d}/{s['n']:3d}={s['p']:.1%} W[{s['lo']:.1%},{s['hi']:.1%}] "
              f"C {s['cyan']:.0%}({s['ncy']}) M {s['mag']:.0%}({s['nmg']}) | distinct {s['distinct']} "
              f"open8 {s['openings']} maxdup {s['maxdup']} n_eff {s['neff']:.0f} "
              f"W_eff[{s['elo']:.1%},{s['ehi']:.1%}] | post-warmup boards {s['bdistinct']}/{s['nhb']} "
              f"n_eff_b {s['neff_b']:.0f} incons {s['incons']} | cpu {s['cpu']:.2f}s act {s['act']:.0f}")
