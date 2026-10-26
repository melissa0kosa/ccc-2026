"""Texel-style fit of value weights (+ policy prior params) on self-play data.
usage: fit.py out.json data1.pkl [data2.pkl ...] [--l2 1e-3] [--pol]"""
import json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
torch.set_num_threads(4)
import hand


def load(paths):
    X, Z, IDX, CNT = [], [], [], []
    for p in paths:
        for rec in pickle.load(open(p, "rb")):
            planes, sc, z, idx, cnt = rec[:5]
            x = np.empty((16, 8, 8), np.float32)
            x[:13] = planes
            x[13:] = sc[:, None, None]
            X.append(x); Z.append(z); IDX.append(idx); CNT.append(cnt)
    return np.stack(X), np.array(Z, np.float32), IDX, CNT


def features(X, bs=4096):
    net = hand.HandNet(hand.n_features()); net.pp = hand.DEFAULT_POL
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            f, _ = net.features(torch.from_numpy(X[i:i + bs]))
            out.append(f)
    return torch.cat(out)


def fit_value(Fe, Z, w0, l2, tr, va):
    w = torch.tensor(np.asarray(w0, np.float32), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    y = (torch.from_numpy(Z) + 1) / 2
    opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")
    def loss_fn(idx):
        lin = Fe[idx] @ w + b
        return torch.nn.functional.binary_cross_entropy_with_logits(2 * lin, y[idx])
    def closure():
        opt.zero_grad()
        l = loss_fn(tr) + l2 * (w ** 2).sum()
        l.backward()
        return l
    with torch.no_grad():
        print(f"init: train {loss_fn(tr):.4f} val {loss_fn(va):.4f}")
    opt.step(closure)
    with torch.no_grad():
        print(f"fit : train {loss_fn(tr):.4f} val {loss_fn(va):.4f}")
        acc = ((Fe[va] @ w + b > 0).float() == y[va]).float().mean()
        print(f"val acc {acc:.3f}")
    return w.detach().numpy(), float(b.detach())


def fit_policy(X, IDX, CNT, pol0, n=20000, steps=150):
    net = hand.HandNet(hand.n_features())
    sel = np.random.default_rng(0).choice(len(X), min(n, len(X)), replace=False)
    Xs = torch.from_numpy(X[sel])
    params = {k: torch.tensor(float(v), requires_grad=True) for k, v in pol0.items()}
    net.pp = params
    with torch.no_grad():
        _, m = net.features(Xs)
    m = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in m.items()}
    rows, cols, tgt = [], [], []
    for r, i in enumerate(sel):
        c = CNT[i].astype(np.float32); c = c / max(c.sum(), 1)
        rows.append(np.full(len(IDX[i]), r)); cols.append(IDX[i].astype(np.int64)); tgt.append(c)
    rows = torch.from_numpy(np.concatenate(rows)); cols = torch.from_numpy(np.concatenate(cols))
    tgt = torch.from_numpy(np.concatenate(tgt))
    opt = torch.optim.Adam(list(params.values()), lr=0.05)
    for s in range(steps):
        logits = net.policy(m)
        l = logits[rows, cols]
        # segment log-softmax over legal moves per row
        mx = torch.zeros(len(sel)).index_reduce_(0, rows, l.detach(), "amax", include_self=False)
        e = torch.exp(l - mx[rows])
        lse = torch.log(torch.zeros(len(sel)).index_add_(0, rows, e)) + mx
        loss = -(tgt * (l - lse[rows])).sum() / len(sel)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 50 == 0 or s == steps - 1:
            print(f"pol step {s} CE {loss.item():.4f}")
    return {k: float(v.detach()) for k, v in params.items()}


if __name__ == "__main__":
    args = sys.argv[1:]
    out = args[0]
    l2 = 1e-4
    do_pol = "--pol" in args
    if "--l2" in args:
        l2 = float(args[args.index("--l2") + 1])
    paths = [a for a in args[1:] if a.endswith(".pkl")]
    X, Z, IDX, CNT = load(paths)
    print("positions", len(X))
    Fe = features(X)
    names = hand.FEATURES
    w0 = np.zeros(len(names), np.float32)
    for k, v in hand.V1_WEIGHTS.items():
        w0[names.index(k)] = v
    n = len(X); cut = int(n * 0.85)
    tr = torch.arange(cut); va = torch.arange(cut, n)
    w, b = fit_value(Fe, Z, w0, l2, tr, va)
    # refit on all
    w, b = fit_value(Fe, Z, w, l2, torch.arange(n), va)
    res = {"weights": {k: float(v) for k, v in zip(names, w)}, "bias": b}
    if do_pol:
        res["pol"] = fit_policy(X, IDX, CNT, hand.DEFAULT_POL)
        print(res["pol"])
    for k, v in sorted(res["weights"].items(), key=lambda t: -abs(t[1]))[:40]:
        print(f"  {k:18s} {v:+.3f}  (v1 {hand.V1_WEIGHTS.get(k, 0):+.2f})")
    Path(out).write_text(json.dumps(res, indent=1))
