"""Train the student. uv run python levels/level4/agents/distill/train.py OUT_NAME "glob1,glob2" [ch] [blocks] [epochs] [plain|sg] [init.pt]"""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
L4 = HERE.parents[1]
sys.path.insert(0, str(L4))
sys.path.insert(0, str(HERE))

T_POL = float(__import__("os").environ.get("T_POL", "0.5"))  # policy temperature on teacher scores (pawn units)
V_SCALE = 3.0    # tanh(best / V_SCALE)
W_Z = float(__import__("os").environ.get("W_Z", "0.25"))  # weight of game outcome in value target
import os
USE_CH = os.environ.get("USE_CH", "1") == "1"


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class Net(nn.Module):
    def __init__(self, c=32, blocks=4, d=16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(16, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.body = nn.Sequential(*[Block(c) for _ in range(blocks)])
        self.q = nn.Conv2d(c, d, 1)
        self.k = nn.Conv2d(c, d, 1)
        self.toh = nn.Conv2d(c, 64, 1)   # channel = from square, spatial = to square
        self.d = d
        self.vconv = nn.Sequential(nn.Conv2d(c, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        self.vfc = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        h = self.body(self.stem(x))
        q = self.q(h).reshape(-1, self.d, 64)              # (B,d,64 from)
        k = self.k(h).reshape(-1, self.d, 64)              # (B,d,64 to)
        bil = torch.matmul(q.transpose(1, 2), k) * (1.0 / self.d ** 0.5)  # (B,64 from,64 to)
        pol = (bil.reshape(-1, 4096) + self.toh(h).flatten(1)).reshape(-1, 4096)
        v = self.vfc(self.vconv(h).flatten(1))
        return pol, torch.tanh(v).reshape(-1)


class SGFeat(nn.Module):
    """Fixed-weight stopgap evaluator (copied forward): returns prior logits, pre-tanh value, feature maps."""
    def __init__(self):
        super().__init__()
        from stopgap import Stopgap
        self.sg = Stopgap()
        for p in self.sg.parameters():
            p.requires_grad = False

    def forward(self, x):
        sg = self.sg
        c = sg.c
        p = sg.pick(x)
        own_k, en_k, locked, urg = p[:, 0:1], p[:, 1:2], p[:, 2:3], p[:, 3:4]
        own_val, en_val, own_nk, en_nk = p[:, 4:5], p[:, 5:6], p[:, 6:7], p[:, 7:8]
        a = sg.adj(torch.cat([locked, own_nk, en_nk], 1))
        danger = torch.clamp(a[:, 0:1] + sg.ring0, 0, 1) * (1 - locked)
        own_guard = torch.clamp(a[:, 1:2], 0, 1)
        en_guard = torch.clamp(a[:, 2:3], 0, 1)
        r = sg.reach(x)
        own_att, en_att = torch.clamp(r[:, 0:1], 0, 1), torch.clamp(r[:, 1:2], 0, 1)

        def s(t):
            return t.flatten(1).sum(1)
        v = c["mat"] * s(own_val - en_val)
        v = v - c["w_danger"] * s(urg * danger * (own_k - en_k))
        v = v + c["w_center"] * s(sg.center * (own_k - en_k))
        v = v + c["w_guard"] * s(own_k * own_guard - en_k * en_guard)
        v = v - c["w_thr"] * s(own_k * en_att * (1 - own_guard)) + c["w_thr"] * s(en_k * own_att * (1 - en_guard))
        v = v + c["w_hang"] * c["mat"] * s(en_val * own_att - 0.5 * own_val * en_att)
        to_term = c["prior_cap"] * en_val - 0.5 * danger * urg
        from_term = c["prior_escape"] * own_k * danger * urg
        policy = (from_term.reshape(-1, 64, 1) + to_term.reshape(-1, 1, 64)).reshape(-1, 4096)
        feats = torch.cat([danger * urg, own_guard, en_guard, own_att, en_att, r[:, 0:1] * 0.25, r[:, 1:2] * 0.25], 1)
        return policy, v.reshape(-1, 1), feats


class NetSG(nn.Module):
    """Residual student on top of the stopgap evaluator (starts equal to stopgap)."""
    def __init__(self, c=32, blocks=4, d=16):
        super().__init__()
        self.f = SGFeat()
        self.stem = nn.Sequential(nn.Conv2d(23, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.body = nn.Sequential(*[Block(c) for _ in range(blocks)])
        self.q = nn.Conv2d(c, d, 1)
        self.k = nn.Conv2d(c, d, 1)
        self.toh = nn.Conv2d(c, 64, 1)
        self.d = d
        self.vconv = nn.Sequential(nn.Conv2d(c, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        self.vfc = nn.Sequential(nn.Linear(256 + 1, 64), nn.ReLU(), nn.Linear(64, 1))
        self.a = nn.Linear(1, 1)
        with torch.no_grad():
            for m in (self.k, self.toh, self.vfc[2]):
                m.weight.zero_()
                m.bias.zero_()
            self.a.weight.fill_(1.0)
            self.a.bias.zero_()
        self.ps = nn.Parameter(torch.ones(1))

    def forward(self, x):
        sp, sv, feats = self.f(x)
        h = self.body(self.stem(torch.cat([x, feats], 1)))
        q = self.q(h).reshape(-1, self.d, 64)
        k = self.k(h).reshape(-1, self.d, 64)
        bil = torch.matmul(q.transpose(1, 2), k) * (1.0 / self.d ** 0.5)
        pol = (bil.reshape(-1, 4096) + self.toh(h).flatten(1) + sp * self.ps).reshape(-1, 4096)
        v = self.a(sv) + self.vfc(torch.cat([self.vconv(h).flatten(1), sv], 1))
        return pol, torch.tanh(v).reshape(-1)


def _hand_base(wname):
    """FastNet (hand evaluator) variant returning (policy logits, pre-tanh lin, scalar features S)."""
    import inspect
    import json
    import textwrap
    hand_dir = L4 / "agents" / "hand"
    if str(hand_dir) not in sys.path:
        sys.path.insert(0, str(hand_dir))
    if wname.startswith("w_v4"):
        import fast4 as fast
        fast.FastNet = fast.FastNet4
    else:
        import fast
    s = textwrap.dedent(inspect.getsource(fast.FastNet.forward))
    s = s.replace("v = torch.tanh(lin)", "v = lin.reshape(-1, 1)")
    s = s.replace("return logits.reshape(-1, 4096), v", "return logits.reshape(-1, 4096), v, S")
    ns = {}
    exec(s, fast.__dict__, ns)
    cls = type("FastLin", (fast.FastNet,), {"forward": ns["forward"]})
    w = json.loads((hand_dir / "data" / f"{wname}.json").read_text())
    base = cls(w["weights"], w.get("bias", 0.0), w.get("pol"))
    for p in base.parameters():
        p.requires_grad = False
    return base


class NetHB(nn.Module):
    """Residual student on top of the hand FastNet evaluator (starts equal to it)."""
    def __init__(self, c=32, blocks=4, d=16, wname="w_v3", use_sg=True):
        super().__init__()
        self.base = _hand_base(wname)
        self.use_sg = use_sg
        if use_sg:
            self.f = SGFeat()
        with torch.no_grad():
            _, _, S = self.base.eval()(torch.zeros(2, 16, 8, 8))
        ns = S.shape[1]
        self.sbn = nn.BatchNorm1d(ns)
        self.stem = nn.Sequential(nn.Conv2d(23 if use_sg else 16, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.body = nn.Sequential(*[Block(c) for _ in range(blocks)])
        self.q = nn.Conv2d(c, d, 1)
        self.k = nn.Conv2d(c, d, 1)
        self.toh = nn.Conv2d(c, 64, 1)
        self.d = d
        self.vconv = nn.Sequential(nn.Conv2d(c, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        self.vfc = nn.Sequential(nn.Linear(256 + 1 + ns, 64), nn.ReLU(), nn.Linear(64, 1))
        self.a = nn.Linear(1, 1)
        with torch.no_grad():
            for m in (self.k, self.toh, self.vfc[2]):
                m.weight.zero_()
                m.bias.zero_()
            self.a.weight.fill_(1.0)
            self.a.bias.zero_()
        self.ps = nn.Parameter(torch.ones(1))

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, x):
        bp, bv, S = self.base(x)
        if self.use_sg:
            _, _, feats = self.f(x)
            h = self.body(self.stem(torch.cat([x, feats], 1)))
        else:
            h = self.body(self.stem(x))
        q = self.q(h).reshape(-1, self.d, 64)
        k = self.k(h).reshape(-1, self.d, 64)
        bil = torch.matmul(q.transpose(1, 2), k) * (1.0 / self.d ** 0.5)
        pol = (bil.reshape(-1, 4096) + self.toh(h).flatten(1) + bp * self.ps).reshape(-1, 4096)
        v = self.a(bv) + self.vfc(torch.cat([self.vconv(h).flatten(1), bv, self.sbn(S)], 1))
        return pol, torch.tanh(v).reshape(-1)


def make_net(arch, ch, blocks):
    if arch.startswith("hf"):
        return NetHB(ch, blocks, wname="w_v4f", use_sg=not arch.startswith("hfs"))
    if arch.startswith("hb"):
        return NetHB(ch, blocks, wname=arch.split(":")[1] if ":" in arch else "w_v3", use_sg=not arch.startswith("hbs"))
    return NetSG(ch, blocks) if arch == "sg" else Net(ch, blocks)


def antidiag_perm():
    p = np.zeros(64, np.int64)
    for r in range(8):
        for c in range(8):
            p[r * 8 + c] = (7 - c) * 8 + (7 - r)
    return p


def load(globs):
    files = []
    for g in globs.split(","):
        files += sorted(glob.glob(str(HERE / "data" / g)))
    parts = [np.load(f) for f in files]
    d = {k: np.concatenate([p[k] for p in parts]) for k in ["obs", "li", "sc", "best", "z"]}
    cp = [p for p in parts if "ch_obs" in p.files]
    if cp:
        d["ch_obs"] = np.concatenate([p["ch_obs"] for p in cp])
        d["ch_v"] = np.concatenate([p["ch_v"] for p in cp])
        print(f"  + {len(d['ch_obs'])} child value samples", flush=True)
    print(f"loaded {len(d['obs'])} positions from {len(files)} files", flush=True)
    return d


def targets(d):
    li = d["li"].astype(np.int64)
    sc = d["sc"].astype(np.float64)
    m = li >= 0
    s = np.where(m, np.clip(sc, -30, 30) / T_POL, -1e9)
    s = s - s.max(1, keepdims=True)
    p = np.exp(s) * m
    p /= p.sum(1, keepdims=True)
    v = (1 - W_Z) * np.tanh(d["best"] / V_SCALE) + W_Z * d["z"]
    return li, m, p.astype(np.float32), v.astype(np.float32)


def main():
    out = sys.argv[1]
    globs = sys.argv[2]
    ch = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    blocks = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    arch = sys.argv[6] if len(sys.argv) > 6 else "plain"
    init = sys.argv[7] if len(sys.argv) > 7 else None
    torch.manual_seed(0)
    dev = "cuda"
    d = load(globs)
    li, m, p, v = targets(d)
    obs = d["obs"].astype(np.float32)
    N = len(obs)
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    nval = max(min(500, N // 5), N // 20)
    vi, ti = perm[:nval], perm[nval:]
    if "ch_obs" in d and USE_CH:
        nc = len(d["ch_obs"])
        obs = np.concatenate([obs, d["ch_obs"].astype(np.float32) / 8.0])
        li = np.concatenate([li, np.zeros((nc, li.shape[1]), li.dtype)])
        m = np.concatenate([m, np.zeros((nc, m.shape[1]), bool)])
        p = np.concatenate([p, np.zeros((nc, p.shape[1]), p.dtype)])
        v = np.concatenate([v, np.tanh(d["ch_v"] / V_SCALE).astype(np.float32)])
        ti = np.concatenate([ti, np.arange(N, N + nc)])
    # policy-loss weight per position from the teacher's root gap (best - second best)
    QG = os.environ.get("QGAP", "0") == "1"
    W = np.ones(len(obs), np.float32)
    if QG:
        scm = np.where(d["li"] >= 0, d["sc"], -1e9)
        top2 = -np.sort(-scm, 1)[:, :2]
        gap = np.clip(top2[:, 0] - top2[:, 1], 0.0, 100.0)
        gap = np.where(top2[:, 1] < -1e8, 0.0, gap)
        w = np.clip(gap, 0.25, 4.0)
        w = w / w.mean()
        W[:len(w)] = w
        print("qgap weights: mean gap", float(np.mean(gap)), "frac gap<0.25", float(np.mean(gap < 0.25)), flush=True)
    Wt = torch.tensor(W, device=dev)
    X = torch.tensor(obs, device=dev)
    LI = torch.tensor(np.where(m, li, 0), device=dev)
    M = torch.tensor(m, device=dev)
    P = torch.tensor(p, device=dev)
    V = torch.tensor(v, device=dev)
    ap = torch.tensor(antidiag_perm(), device=dev)
    LI_aug = ap[LI // 64] * 64 + ap[LI % 64]

    net = make_net(arch, ch, blocks).to(dev)
    if init:
        net.load_state_dict(torch.load(init))
    bs = 256
    bs = min(bs, max(16, len(ti) // 8))
    steps = epochs * (len(ti) // bs)
    LR = float(os.environ.get("LR", "2e-3"))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.15)
    ti_t = torch.tensor(ti, device=dev)
    vi_t = torch.tensor(vi, device=dev)

    import os as _os
    AUG8 = _os.environ.get("AUG", "2") == "8"
    ops = [lambda r, c: (r, c), lambda r, c: (c, 7 - r), lambda r, c: (7 - r, 7 - c), lambda r, c: (7 - c, r),
           lambda r, c: (c, r), lambda r, c: (7 - c, 7 - r), lambda r, c: (r, 7 - c), lambda r, c: (7 - r, c)]
    PERM = np.zeros((8, 64), np.int64)
    for o, f in enumerate(ops):
        for r in range(8):
            for c in range(8):
                rr, cc = f(r, c)
                PERM[o, r * 8 + c] = rr * 8 + cc
    INV = np.argsort(PERM, 1)
    PERM_t = torch.tensor(PERM, device=dev)
    INV_t = torch.tensor(INV, device=dev)

    def batch_loss(idx, aug):
        x = X[idx]
        li_ = LI[idx]
        if aug is not None and AUG8:
            op = torch.randint(0, 8, (len(idx),), device=dev)
            g = INV_t[op]                                   # new[sq] = old[INV[sq]]
            x = torch.gather(x.reshape(-1, 16, 64), 2, g[:, None, :].expand(-1, 16, 64)).reshape(-1, 16, 8, 8)
            pm = PERM_t[op]
            li_ = torch.gather(pm, 1, li_ // 64) * 64 + torch.gather(pm, 1, li_ % 64)
        elif aug is not None:
            xa = torch.flip(x.transpose(-1, -2), dims=(-1, -2))
            x = torch.where(aug[:, None, None, None], xa, x)
            li_ = torch.where(aug[:, None], LI_aug[idx], li_)
        pol, val = net(x)
        lg = torch.gather(pol, 1, li_)
        lg = lg.masked_fill(~M[idx], -1e9)
        lp = F.log_softmax(lg, 1)
        hp = M[idx].any(1).float()
        wq = Wt[idx] * hp
        pl = -(wq * (P[idx] * lp).sum(1)).sum() / wq.sum().clamp(min=1e-6)
        vl = F.mse_loss(val, V[idx])
        # accuracy: argmax match with teacher best
        acc = (lg.argmax(1) == P[idx].argmax(1)).float().mean()
        return pl, vl, acc

    t0 = time.time()
    step = 0
    for ep in range(epochs):
        net.train()
        sh = ti_t[torch.randperm(len(ti_t), device=dev)]
        tl = []
        for b in range(len(ti) // bs):
            idx = sh[b * bs:(b + 1) * bs]
            aug = torch.rand(len(idx), device=dev) < 0.5
            pl, vl, acc = batch_loss(idx, aug)
            loss = pl + 2.0 * vl
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            tl.append((pl.item(), vl.item()))
        if ep % 3 == 2 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                pl, vl, acc = batch_loss(vi_t, None)
            tlm = np.mean(tl, 0)
            print(f"ep {ep} train pol {tlm[0]:.3f} val {tlm[1]:.3f} | valid pol {pl:.3f} val {vl:.3f} "
                  f"top1 {acc:.3f} | {time.time()-t0:.0f}s", flush=True)
    torch.save(net.state_dict(), HERE / f"{out}.pt")
    from export_util import export as export_onnx
    path = L4 / "models" / f"{out}.onnx"
    export_onnx(net.cpu().eval(), path)
    print("exported", path, path.stat().st_size, "bytes", flush=True)


if __name__ == "__main__":
    main()
