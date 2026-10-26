"""Residual AZ/ExIt on top of the frozen hand_v2 evaluator.

value = tanh(v_h + dv), policy = p_h + dp; dv/dp from a small conv trunk with zero-initialised
output layers, so step 0 == hand_v2 exactly.
Launcher: levels/level4/agents/az/res.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import azlib
import hand_v2_base as hb
from azlib import DATA, HERE, MODELS, SEAT, STOPGAP, Buffer, Res, arena, export, fmt, log, run_jobs, \
    sp_jobs, summarize, train, validate

HAND = MODELS / "hand_v2.onnx"
RCKPT = HERE / "ckpt_res"
RRESULTS = HERE / "RESULTS.md"


def build_base():
    d = json.loads((HERE / "w_v2.json").read_text())
    return hb.build(d["weights"], d.get("bias", 0.0), d.get("pol"))


def build_fast_base(kind="v2"):
    if kind == "v4f":
        import fast4_base
        d = json.loads((HERE / "w_v4f.json").read_text())
        net = fast4_base.FastNet4(d["weights"], d.get("bias", 0.0), d.get("pol")).eval()
    else:
        import fast_base
        d = json.loads((HERE / "w_v2.json").read_text())
        net = fast_base.FastNet(d["weights"], d.get("bias", 0.0), d.get("pol")).eval()
    net.return_parts = True
    return net


class ResFast(nn.Module):
    """Same residual as ResModel, but on the lean FastNet base (hand_v2fast, output-identical to hand_v2).
    The value MLP sees the base's S/G feature vectors (batch-normed) instead of HandNet features."""

    def __init__(self, c=32, blocks=2, k=16, kind="v2"):
        super().__init__()
        self.base = build_fast_base(kind)
        with torch.no_grad():
            _, _, S, G = self.base(torch.zeros(2, 16, 8, 8))
        nf = S.shape[1] + G.shape[1]
        self.k = k
        self.stem = nn.Sequential(nn.Conv2d(16, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.blocks = nn.Sequential(*[Res(c) for _ in range(blocks)])
        self.pf = nn.Conv2d(c, k, 1)
        self.pt = nn.Conv2d(c, k, 1)
        self.p64 = nn.Conv2d(c, 64, 1)
        self.v1 = nn.Sequential(nn.Conv2d(c, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU())
        self.fbn = nn.BatchNorm1d(nf)
        self.vfc1 = nn.Linear(256 + c + nf, 64)
        self.vfc2 = nn.Linear(64, 1)
        for m in (self.pt, self.p64, self.vfc2):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.scale = 1.0 / float(np.sqrt(k))

    def forward(self, x):
        ph, vh, S, G = self.base(x)
        vh = vh.reshape(-1)
        h = self.blocks(self.stem(x))
        f = self.pf(h).reshape(-1, self.k, 64).transpose(1, 2)
        t = self.pt(h).reshape(-1, self.k, 64)
        dp = (torch.matmul(f, t) * self.scale + self.p64(h).reshape(-1, 64, 64)).reshape(-1, 4096)
        z = torch.cat([self.v1(h).flatten(1), h.mean(dim=(2, 3)), self.fbn(torch.cat([S, G], 1))], 1)
        dv = self.vfc2(F.relu(self.vfc1(z))).reshape(-1)
        return ph + dp, torch.tanh(vh + dv)


def make_model(args):
    if getattr(args, "base", "hand") == "fast":
        return ResFast(args.channels, args.blocks)
    if getattr(args, "base", "hand") == "v4f":
        return ResFast(args.channels, args.blocks, kind="v4f")
    return ResModel(args.channels, args.blocks)


def graph_size(path):
    import onnx
    m = onnx.load(str(path))
    return len(m.graph.node), len(m.graph.initializer)


class ResModel(nn.Module):
    def __init__(self, c=32, blocks=3, k=16):
        super().__init__()
        self.base = build_base()
        for p in self.base.parameters():
            p.requires_grad_(False)
        nf = self.base.n_feat
        self.k = k
        self.stem = nn.Sequential(nn.Conv2d(16, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.blocks = nn.Sequential(*[Res(c) for _ in range(blocks)])
        self.pf = nn.Conv2d(c, k, 1)
        self.pt = nn.Conv2d(c, k, 1)
        self.p64 = nn.Conv2d(c, 64, 1)
        self.v1 = nn.Sequential(nn.Conv2d(c, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU())
        self.vfc1 = nn.Linear(256 + c + nf, 64)
        self.vfc2 = nn.Linear(64, 1)
        for m in (self.pt, self.p64, self.vfc2):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.scale = 1.0 / float(np.sqrt(k))

    def forward(self, x):
        feats, m = self.base.features(x)
        vh = self.base.lin(feats).reshape(-1)
        ph = self.base.policy(m)
        h = self.blocks(self.stem(x))
        f = self.pf(h).reshape(-1, self.k, 64).transpose(1, 2)
        t = self.pt(h).reshape(-1, self.k, 64)
        dp = (torch.matmul(f, t) * self.scale + self.p64(h).reshape(-1, 64, 64)).reshape(-1, 4096)
        z = torch.cat([self.v1(h).flatten(1), h.mean(dim=(2, 3)), feats], 1)
        dv = self.vfc2(F.relu(self.vfc1(z))).reshape(-1)
        return ph + dp, torch.tanh(vh + dv)


def parity(path, n=64):
    """Compare exported ONNX vs hand_v2.onnx on real positions."""
    import onnxruntime as ort
    with open(sorted(DATA.glob("games_teacher_*.pkl"))[0], "rb") as f:
        games = pickle.load(f)
    x = np.concatenate([g["obs"] for g in games[:10]])[:n].astype(np.float32) / 8.0
    a = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"observation": x})
    b = ort.InferenceSession(str(HAND), providers=["CPUExecutionProvider"]).run(None, {"observation": x})
    return float(np.abs(a[0] - b[0]).max()), float(np.abs(a[1] - b[1]).max())


def load_hand_sp(path):
    """agents/hand gen_data.py format -> one pseudo-game dict for Buffer."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    obs = np.zeros((len(data), 16, 8, 8), np.uint8)
    idx, p, z = [], [], np.zeros(len(data), np.float32)
    qv = np.zeros(len(data), np.float32)
    for i, rec in enumerate(data):
        planes, sc, zz, ii, cc = rec[:5]
        qv[i] = rec[5] if len(rec) > 5 else zz
        obs[i, :13] = planes * 8
        obs[i, 13:] = np.rint(np.asarray(sc, np.float32) * 8).astype(np.uint8)[:, None, None]
        idx.append(np.asarray(ii, np.int16))
        c = np.asarray(cc, np.float32)
        p.append(c / max(1.0, c.sum()))
        z[i] = zz
    return dict(obs=obs, idx=idx, p=p, z=z, q=qv, cyan_won=False, n=len(data), cpu=0.0)


def append_row(row):
    if not RRESULTS.exists():
        RRESULTS.write_text("")
    txt = RRESULTS.read_text()
    if "## Residual" not in txt:
        with open(RRESULTS, "a") as f:
            f.write("\n## Residual on hand_v2 (az_res_vN): value=tanh(v_h+dv), policy=p_h+dp, 32ch x 3 blocks\n\n"
                    "| model | vs hand_v2 | vs stopgap_v1 | vs seat_l3_material | notes |\n|---|---|---|---|---|\n")
    with open(RRESULTS, "a") as f:
        f.write(row + "\n")


def evaluate(cur, args, ver, best, full):
    """Screen vs the reference (az_res_v6) head-to-head; extend promising ones to 300 games.
    CANDIDATE = beats the reference with Wilson 95% lower bound > 0.5."""
    from c1 import wilson
    ref = Path(args.ref)
    r = arena(cur, ref, args.gate_games, workers=args.workers, verbose=False)
    wins, games = r["wins"], r["games"]
    cyan_w = r["cyan_rate"] * (games / 2)
    note = ""
    if r["rate"] >= args.extend_at:
        r2 = arena(cur, ref, args.confirm_games, workers=args.workers, verbose=False)
        wins += r2["wins"]
        cyan_w += r2["cyan_rate"] * (r2["games"] / 2)
        games += r2["games"]
    lo, hi = wilson(wins, games)
    h2h = f"{wins}/{games}={wins / games:.0%} [{lo:.0%},{hi:.0%}] C{cyan_w / (games / 2):.0%} {r['cpu_per_game']:.1f}s"
    if lo > 0.5:
        ok, msg = validate(cur)
        note = (f"**CANDIDATE: beats {ref.stem} head-to-head, Wilson LB {lo:.2f} > 0.5** "
                f"(local validator {'ACCEPTED' if ok else 'REJECTED - local check is wall-time flaky under load, re-validate'})")
        if wins / games > best[1]:
            best = (str(cur), wins / games)
    r_hand = arena(cur, HAND, 40, workers=args.workers, verbose=False) if full else None
    log(f"res v{ver} eval: vs {ref.stem} {h2h} | hand_v2 {fmt(r_hand)} | graph {graph_size(cur)} {note}")
    append_row(f"| az_res_v{ver} | {fmt(r_hand)} | - | - | h2h vs {ref.stem}: {h2h}; {note} |")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--gens", type=int, default=200)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--mix-games", type=int, default=100)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--window", type=int, default=120_000)
    ap.add_argument("--reuse", type=float, default=4.0)
    ap.add_argument("--gate-games", type=int, default=60)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--qmix", type=float, default=0.3)
    ap.add_argument("--boot-steps", type=int, default=800)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--start-version", type=int, default=0)
    ap.add_argument("--full-every", type=int, default=3)
    ap.add_argument("--no-hand-data", action="store_true")
    ap.add_argument("--base", type=str, default="hand")
    ap.add_argument("--eval-first", action="store_true")
    ap.add_argument("--ref", type=str, default=str(MODELS / "az_res_v6.onnx"))
    ap.add_argument("--extend-at", type=float, default=0.58)
    ap.add_argument("--confirm-games", type=int, default=240)
    ap.add_argument("--mix-opp", type=str, default=str(HAND))
    ap.add_argument("--init", type=str, default="")
    args = ap.parse_args()

    RCKPT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    dev = "cuda"
    torch.manual_seed(0)
    net = make_model(args).to(dev)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    buf = Buffer(args.window)
    ver = args.start_version
    best = (str(HAND), 0.5)
    pool = get_context("spawn").Pool(args.workers)
    seed = int(time.time()) % 100000 * 1000 + 500

    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=dev)["net"])
        log("res init from", args.init)
    if args.resume:
        st = torch.load(args.resume, map_location=dev)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        best = tuple(st.get("best", best))
        hsp = HERE.parent / "hand" / "data" / "sp_v2.pkl"
        if hsp.exists() and not args.no_hand_data:
            buf.add(load_hand_sp(hsp))
        for f in sorted(DATA.glob("games_res_*.pkl"), key=lambda q: q.stat().st_mtime):
            with open(f, "rb") as fh:
                for g in pickle.load(fh):
                    buf.add(g)
        buf.trim()
        cur = MODELS / f"az_res_v{ver - 1}.onnx"
        log("res resumed", args.resume, "buf", buf.n(), "best", best)
        if args.eval_first:
            best = evaluate(cur, args, ver - 1, best, full=True)
    else:
        cur = export(net, MODELS / f"az_res_v{ver}.onnx")
        dp, dv = parity(cur)
        log(f"res v{ver} (== hand_v2) parity: max|dpolicy| {dp:.2e} max|dvalue| {dv:.2e}")
        r = arena(cur, HAND, 40, workers=args.workers, verbose=False)
        log(f"res v{ver} vs hand_v2 sanity: {fmt(r)}")
        append_row(f"| az_res_v{ver} | {fmt(r)} | - | - | step 0 == hand_v2 (parity {dp:.1e}/{dv:.1e}) |")
        ver += 1
        # bootstrap from hand_v2 self-play data already on disk
        hsp = HERE.parent / "hand" / "data" / "sp_v2.pkl"
        if hsp.exists():
            g = load_hand_sp(hsp)
            buf.add(g)
            log(f"res: loaded hand sp_v2 {g['n']} positions")
            lp, lv, n = train(net, opt, buf, args.boot_steps, device=dev, qmix=0.0)
            log(f"res v{ver}: boot train {args.boot_steps} steps lpol {lp:.3f} lval {lv:.3f}")
            cur = export(net, MODELS / f"az_res_v{ver}.onnx")
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "best": best}, RCKPT / f"az_res_v{ver}.pt")
            best = evaluate(cur, args, ver, best, full=True)
            ver += 1

    for gen in range(args.gens):
        t = time.time()
        jobs = sp_jobs(cur, cur, args.games, args.sims, seed, 12, 0.3)
        jobs += sp_jobs(cur, args.mix_opp, args.mix_games, args.sims, seed + args.games, 12, 0.3, alternate=True)
        games = run_jobs(pool, jobs, f"res_v{ver}_{seed}")
        seed += 10000
        for g in games:
            buf.add(g)
        buf.trim()
        new = sum(g["n"] for g in games)
        log(f"res gen v{ver}: selfplay+mix {summarize(games)} {time.time() - t:.0f}s buf {buf.n()}")
        steps = int(args.reuse * new / 512)
        lp, lv, n = train(net, opt, buf, steps, device=dev, qmix=args.qmix)
        log(f"res gen v{ver}: train {steps} steps lpol {lp:.3f} lval {lv:.3f}")
        cur = export(net, MODELS / f"az_res_v{ver}.onnx")
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "best": best}, RCKPT / f"az_res_v{ver}.pt")
        best = evaluate(cur, args, ver, best, full=(gen % args.full_every == args.full_every - 1))
        ver += 1
