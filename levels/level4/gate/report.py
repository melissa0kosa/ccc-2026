"""Build the gate tables from gate/games*.jsonl + gate/checks.jsonl. Prints markdown (paste into RECOMMEND.md).
usage: report.py cand1.onnx cand2.onnx ...   (live seat = hand_v2.onnx)"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from summarize import summarize  # noqa: E402

SEAT = "hand_v2.onnx"
by = defaultdict(list)
for f in sorted(HERE.glob("games*.jsonl")):
    for ln in f.read_text().splitlines():
        r = json.loads(ln)
        by[(r["a"], r["b"])].append(r)
checks = {}
if (HERE / "checks.jsonl").exists():
    for ln in (HERE / "checks.jsonl").read_text().splitlines():
        r = json.loads(ln)
        checks[r["name"]] = r


def S(a, b):
    return summarize(by[(a, b)]) if by.get((a, b)) else None


def fmt(s):
    if not s:
        return "not run"
    return (f"{s['w']}/{s['n']} = {s['p']:.1%} [Wilson {s['lo']:.1%}, {s['hi']:.1%}]; "
            f"cyan {s['cyan']:.0%} / magenta {s['mag']:.0%}")


base_sg = S(SEAT, "stopgap_v1.onnx")
lines = []
for c in sys.argv[1:]:
    ck = checks.get(c)
    vs = S(c, SEAT)
    gr = S(c, "greedy")
    sg = S(c, "stopgap_v1.onnx")
    rows = []
    ok_all = True

    def row(name, val, ok):
        global ok_all
        ok_all = ok_all and ok is True
        rows.append(f"| {name} | {val} | {'PASS' if ok is True else ('FAIL' if ok is False else 'pending')} |")

    row("Official validator", f"ACCEPTED (try {ck['tries']})" if ck and ck["validator"] else ("REJECTED" if ck else "not run"),
        (ck["validator"] if ck else None))
    row("Graph caps (nodes<=1024, inits<=512)", f"{ck['nodes']} nodes / {ck['inits']} inits" if ck else "not run",
        ck["caps_ok"] if ck else None)
    row(f">=200 games vs {SEAT}, Wilson lower > 55%", fmt(vs),
        None if not vs or (vs["n"] < 200 and vs["hi"] > 0.55) else (vs["n"] >= 200 and vs["lo"] > 0.55))
    if vs:
        row("Effective sample (distinct games / post-warmup boards)",
            f"{vs['distinct']}/{vs['n']} distinct games, n_eff {vs['neff']:.0f}; post-warmup boards "
            f"{vs['bdistinct']}/{vs['nhb']}; dedup Wilson [{vs['elo']:.1%}, {vs['ehi']:.1%}]",
            None if vs["n"] < 200 else vs["elo"] > 0.55)
    row("Neither colour < 50% (vs seat)", f"cyan {vs['cyan']:.0%} ({vs['ncy']}) / magenta {vs['mag']:.0%} ({vs['nmg']})" if vs else "not run",
        None if not vs else min(vs["cyan"], vs["mag"]) >= 0.5)
    row(">=95% vs greedy (20 games)", fmt(gr), None if not gr else (gr["p"] >= 0.95 and gr["n"] >= 20))
    if sg and base_sg:
        row("vs stopgap_v1, no drop >10 pp vs hand_v2's own",
            f"{fmt(sg)}; hand_v2: {base_sg['w']}/{base_sg['n']} = {base_sg['p']:.1%}; delta {100 * (sg['p'] - base_sg['p']):+.1f} pp",
            sg["p"] >= base_sg["p"] - 0.10)
    else:
        row("vs stopgap_v1, no drop >10 pp vs hand_v2's own", fmt(sg) + (f"; hand_v2 {fmt(base_sg)}" if base_sg else ""), None)
    row("CPU/game, both sides (<= ~3 s)", f"{vs['cpu']:.2f} s (vs seat), {vs['act']:.0f} actions" if vs else "not run",
        None if not vs else vs["cpu"] <= 3.0)
    lines.append(f"### {c}: {'PASS' if ok_all else 'not passed'}\n\n| criterion | measured | result |\n|---|---|---|\n"
                 + "\n".join(rows) + "\n")
print("\n".join(lines))
other = [(a, b) for (a, b) in by if a not in sys.argv[1:]]
for a, b in other:
    print(f"- {a} vs {b}: {fmt(S(a, b))}")
