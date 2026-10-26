"""Texel fit of v5 value weights (FastNet5 features) with search-blended targets.
target = 0.7*q_root + 0.3*z where root Q is recorded (sp_v4f), else z. L2 toward init weights.
usage: fit5.py out.json init.json data1.pkl [...] [--l2 1e-4] [--alpha 0.7]"""
import json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
torch.set_num_threads(4)
import fast5


def load(paths, alpha):
    X, T = [], []
    for p in paths:
        for rec in pickle.load(open(p, "rb")):
            planes, sc, z = rec[0], rec[1], rec[2]
            x = np.empty((16, 8, 8), np.float32); x[:13] = planes; x[13:] = sc[:, None, None]
            X.append(x)
            T.append(alpha * rec[5] + (1 - alpha) * z if len(rec) > 5 else z)
    return np.stack(X), np.array(T, np.float32)


def fit_value(Fe, T, w0, l2, tr, va):
    w = torch.tensor(w0.copy(), requires_grad=True); b = torch.zeros(1, requires_grad=True)
    w0t = torch.tensor(w0)
    y = (torch.from_numpy(T) + 1) / 2
    opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")
    lf = lambda idx: torch.nn.functional.binary_cross_entropy_with_logits(2 * (Fe[idx] @ w + b), y[idx])
    def closure():
        opt.zero_grad(); l = lf(tr) + l2 * ((w - w0t) ** 2).sum(); l.backward(); return l
    with torch.no_grad():
        print(f"init: train {lf(tr):.4f} val {lf(va):.4f}")
    opt.step(closure)
    with torch.no_grad():
        acc = (((Fe[va] @ w + b) > 0).float() == (y[va] > 0.5).float()).float().mean()
        print(f"fit : train {lf(tr):.4f} val {lf(va):.4f} val sign-acc {acc:.3f}")
    return w.detach().numpy(), float(b.detach())


if __name__ == "__main__":
    args = sys.argv[1:]
    out, init = args[0], json.loads(Path(args[1]).read_text())
    l2 = float(args[args.index("--l2") + 1]) if "--l2" in args else 1e-4
    alpha = float(args[args.index("--alpha") + 1]) if "--alpha" in args else 0.7
    paths = [a for a in args[2:] if a.endswith(".pkl")]
    X, T = load(paths, alpha)
    print("positions", len(X))
    net = fast5.FastNet5({}, 0.0, init.get("pol")).eval()
    with torch.no_grad():
        Fe = torch.cat([net.features(torch.from_numpy(X[i:i + 4096])) for i in range(0, len(X), 4096)])
    names = net.names
    w0 = np.array([init["weights"].get(n, 0.0) for n in names], np.float32)
    n = len(X); cut = int(n * 0.85)
    w, b = fit_value(Fe, T, w0, l2, torch.arange(cut), torch.arange(cut, n))   # report this val
    w, b = fit_value(Fe, T, w, l2, torch.arange(n), torch.arange(cut, n))
    res = {"weights": {k: float(v) for k, v in zip(names, w)}, "bias": b, "pol": init.get("pol")}
    for k in fast5.EXTRA:
        print(f"  {k:18s} {res['weights'][k]:+.3f}")
    Path(out).write_text(json.dumps(res, indent=1))
