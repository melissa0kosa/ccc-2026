"""Texel fit of the v4 value weights (FastNet4 features).
usage: fit4.py out.json init.json data1.pkl [...] [--l2 1e-4]"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
torch.set_num_threads(4)
import fit, fast4

if __name__ == "__main__":
    args = sys.argv[1:]
    out, init = args[0], json.loads(Path(args[1]).read_text())
    l2 = float(args[args.index("--l2") + 1]) if "--l2" in args else 1e-4
    paths = [a for a in args[2:] if a.endswith(".pkl")]
    X, Z, IDX, CNT = fit.load(paths)
    print("positions", len(X))
    net = fast4.FastNet4({}, 0.0, init.get("pol")).eval()
    with torch.no_grad():
        Fe = torch.cat([net.features(torch.from_numpy(X[i:i + 4096])) for i in range(0, len(X), 4096)])
    names = net.names
    w0 = np.array([init["weights"].get(n, 0.0) for n in names], np.float32)
    n = len(X); cut = int(n * 0.85)
    tr = torch.arange(cut); va = torch.arange(cut, n)
    w, b = fit.fit_value(Fe, Z, w0, l2, tr, va)
    w, b = fit.fit_value(Fe, Z, w, l2, torch.arange(n), va)
    res = {"weights": {k: float(v) for k, v in zip(names, w)}, "bias": b, "pol": init.get("pol")}
    for k in fast4.EXTRA:
        print(f"  {k:18s} {res['weights'][k]:+.3f}")
    for k, v in sorted(res["weights"].items(), key=lambda t: -abs(t[1]))[:15]:
        print(f"  {k:18s} {v:+.3f}  (init {init['weights'].get(k, 0):+.3f})")
    Path(out).write_text(json.dumps(res, indent=1))
