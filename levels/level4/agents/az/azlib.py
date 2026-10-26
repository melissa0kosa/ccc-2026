"""AlphaZero-style self-play loop for COLONY: ONE (Level 4).

Run from repo root:
    uv run python levels/level4/agents/az/az.py [--resume]
Writes models to levels/level4/models/az_vN.onnx, logs to agents/az/log.txt.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
L4 = HERE.parents[1]
sys.path.insert(0, str(L4))
sys.path.insert(0, str(HERE))

from c1 import MODELS, arena, export_onnx, validate  # noqa: E402

CKPT = HERE / "ckpt"
DATA = HERE / "data"
LOG = HERE / "log.txt"
RESULTS = HERE / "RESULTS.md"
SEAT = MODELS / "seat_l3_material.onnx"
STOPGAP = MODELS / "stopgap_v1.onnx"


def log(*a):
    s = time.strftime("%H:%M:%S", time.gmtime()) + " " + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")


# ----------------------------------------------------------------------------- net

class Res(nn.Module):
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
    def __init__(self, c=32, blocks=4, k=32):
        super().__init__()
        self.k = k
        self.stem = nn.Sequential(nn.Conv2d(16, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.blocks = nn.Sequential(*[Res(c) for _ in range(blocks)])
        self.pf = nn.Conv2d(c, k, 1)
        self.pt = nn.Conv2d(c, k, 1)
        self.p64 = nn.Conv2d(c, 64, 1)
        self.v1 = nn.Sequential(nn.Conv2d(c, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU())
        self.vfc1 = nn.Linear(256 + c, 64)
        self.vfc2 = nn.Linear(64, 1)
        self.scale = 1.0 / math.sqrt(k)

    def forward(self, x):
        h = self.blocks(self.stem(x))
        f = self.pf(h).reshape(-1, self.k, 64).transpose(1, 2)  # (B, from, k)
        t = self.pt(h).reshape(-1, self.k, 64)  # (B, k, to)
        pol = (torch.matmul(f, t) * self.scale + self.p64(h).reshape(-1, 64, 64)).reshape(-1, 4096)
        v = torch.cat([self.v1(h).flatten(1), h.mean(dim=(2, 3))], 1)
        v = torch.tanh(self.vfc2(F.relu(self.vfc1(v)))).reshape(-1)
        return pol, v


# ----------------------------------------------------------------------------- data

def _flip_table():
    t = np.zeros(4096, np.int64)
    for f in range(64):
        fr, fc = divmod(f, 8)
        ff = fr * 8 + 7 - fc
        for to in range(64):
            tr, tc = divmod(to, 8)
            t[f * 64 + to] = ff * 64 + tr * 8 + 7 - tc
    return t


FLIP = _flip_table()


def _d4():
    P = np.zeros((8, 64), np.int64)
    for k in range(8):
        for sq in range(64):
            r, c = divmod(sq, 8)
            if k & 4:
                r, c = c, r
            if k & 1:
                c = 7 - c
            if k & 2:
                r = 7 - r
            P[k, sq] = r * 8 + c
    inv = np.argsort(P, axis=1)
    idx = (P[:, :, None] * 64 + P[:, None, :]).reshape(8, 4096)
    return inv, idx


D4_INV, D4_IDX = _d4()  # validated by symcheck.py (12192 position x symmetry checks, 0 mismatches)
SYMS = 8


class Buffer:
    def __init__(self, max_samples):
        self.games = []
        self.max = max_samples

    def add(self, g):
        self.games.append(g)

    def n(self):
        return sum(g["n"] for g in self.games)

    def trim(self):
        while self.n() > self.max and len(self.games) > 1:
            self.games.pop(0)

    def tensors(self, device, qmix=0.0):
        obs = np.concatenate([g["obs"] for g in self.games])
        z = np.concatenate([g["z"] for g in self.games])
        q = np.concatenate([g["q"] for g in self.games])
        idx_l = [a for g in self.games for a in g["idx"]]
        p_l = [a for g in self.games for a in g["p"]]
        L = max(len(a) for a in idx_l)
        idx = np.full((len(idx_l), L), -1, np.int64)
        p = np.zeros((len(idx_l), L), np.float32)
        for i, (a, b) in enumerate(zip(idx_l, p_l)):
            idx[i, :len(a)] = a
            p[i, :len(b)] = b
        zt = (1 - qmix) * z + qmix * q
        if getattr(self, "lowmat_outcome", False):
            # outcome-only value where each side has <= 2 non-Kernel pieces (search Q shares the endgame blind spot)
            own = obs[:, 0:5].reshape(len(obs), -1).sum(1) / 8.0
            en = obs[:, 6:11].reshape(len(obs), -1).sum(1) / 8.0
            low = (own <= 2) & (en <= 2)
            zt = np.where(low, z, zt).astype(np.float32)
            self.low_frac = float(low.mean())
            self._vw = np.where((own <= 1) & (en <= 1), 2.0, 1.0).astype(np.float32)  # value loss x2 deep endgame
        return (torch.from_numpy(obs).to(device), torch.from_numpy(idx).to(device),
                torch.from_numpy(p).to(device), torch.from_numpy(zt).to(device))


def train(net, opt, buf, steps, batch=512, device="cuda", qmix=0.0, sched=None):
    obs, idx, p, z = buf.tensors(device, qmix)
    vw = torch.from_numpy(buf._vw).to(device) if getattr(buf, "_vw", None) is not None else None
    inv = torch.from_numpy(D4_INV).to(device)
    imap = torch.from_numpy(D4_IDX).to(device)
    N = obs.shape[0]
    net.train()
    lp_sum = lv_sum = 0.0
    for s in range(steps):
        b = torch.randint(0, N, (batch,), device=device)
        x = obs[b].float() / 8.0
        bi = idx[b]
        pad = bi < 0
        bi = bi.clamp(min=0)
        sy = torch.randint(0, SYMS, (batch,), device=device)
        x = x.reshape(batch, 16, 64).gather(2, inv[sy][:, None, :].expand(batch, 16, 64)).reshape(batch, 16, 8, 8)
        bi = imap[sy].gather(1, bi)
        logits, v = net(x)
        lg = logits.gather(1, bi).masked_fill(pad, -1e9)
        logp = F.log_softmax(lg, 1).masked_fill(pad, 0.0)
        lpol = -(p[b] * logp).sum(1).mean()
        lval = F.mse_loss(v, z[b]) if vw is None else (vw[b] * (v - z[b]) ** 2).sum() / vw[b].sum()
        loss = lpol + lval
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        lp_sum += lpol.item() if s % 20 == 0 else 0
        lv_sum += lval.item() if s % 20 == 0 else 0
    k = max(1, (steps + 19) // 20)
    return lp_sum / k, lv_sum / k, N


# ----------------------------------------------------------------------------- loop

def run_jobs(pool, jobs, tag):
    out = []
    from sp import selfplay_game
    for r in pool.imap_unordered(selfplay_game, jobs, chunksize=4):
        if r is not None:
            out.append(r)
    import pickle
    with open(DATA / f"games_{tag}.pkl", "wb") as f:
        pickle.dump(out, f)
    return out


def sp_jobs(cyan, mag, games, sims, seed, temp_moves, late_temp, alternate=False):
    jobs = []
    for i in range(games):
        a, b = (cyan, mag) if (not alternate or i % 2 == 0) else (mag, cyan)
        jobs.append((str(a), str(b), sims, seed + i, temp_moves, late_temp))
    return jobs


def summarize(games):
    return (f"{len(games)} games {sum(g['n'] for g in games)} samples "
            f"cyan_won {np.mean([g['cyan_won'] for g in games]):.2f} "
            f"len {np.mean([g['n'] for g in games]):.1f} cpu/game {np.mean([g['cpu'] for g in games]):.2f}")


def _sp(args):
    from sp import selfplay_game
    return selfplay_game(args)


def export(net, path, check=False):
    import onnx
    m = copy.deepcopy(net).cpu().eval()
    export_onnx(m, path, check=False)
    om = onnx.load(str(path))
    for o in om.graph.output:
        dims = o.type.tensor_type.shape.dim
        dims[0].dim_param = "batch"
        if o.name == "policy":
            dims[1].dim_value = 4096
    onnx.checker.check_model(om)
    onnx.save(om, str(path))
    if check:
        ok, msg = validate(path)
        if not ok:
            raise RuntimeError(msg)
    return Path(path)


def append_result(row: str):
    if not RESULTS.exists():
        RESULTS.write_text(
            "# AZ self-play results (100 sims, arena games alternate colours)\n\n"
            "Format: wins/games=rate [95% Wilson CI] C=win rate as Cyan / M=as Magenta, cpu s/game (both sides)\n\n"
            "| model | vs seat_l3_material | vs stopgap_v1 | vs greedy | vs prev best | notes |\n"
            "|---|---|---|---|---|---|\n")
    with open(RESULTS, "a") as f:
        f.write(row + "\n")


def fmt(r):
    if r is None:
        return "-"
    return (f"{r['wins']}/{r['games']}={r['rate']:.0%} [{r['ci'][0]:.0%},{r['ci'][1]:.0%}] "
            f"C{r['cyan_rate']:.0%}/M{r['magenta_rate']:.0%} {r['cpu_per_game']:.1f}s")


def evaluate(cur, best, args, ver, full=True):
    """best = (path, stopgap_rate) or None. Promote on higher win rate vs stopgap_v1."""
    r_stop = arena(cur, STOPGAP, args.stop_games, workers=args.workers, verbose=False)
    r_seat = r_greedy = r_best = None
    if full:
        r_seat = arena(cur, SEAT, args.eval_games, workers=args.workers, verbose=False)
        r_greedy = arena(cur, "greedy", 20, workers=args.workers, verbose=False)
        if best is not None:
            r_best = arena(cur, best[0], args.eval_games, workers=args.workers, verbose=False)
    note = ""
    if best is None or r_stop["rate"] > best[1]:
        ok, msg = validate(cur)
        if ok:
            best = (str(cur), r_stop["rate"])
            note = "NEW BEST vs stopgap (validated)"
        else:
            note = "validator REJECTED"
            log(msg)
    log(f"v{ver} eval: STOPGAP {fmt(r_stop)} | seat {fmt(r_seat)} | greedy {fmt(r_greedy)} | prev-best {fmt(r_best)} {note}")
    append_result(f"| az_v{ver} | {fmt(r_seat)} | {fmt(r_stop)} | {fmt(r_greedy)} | {fmt(r_best)} | {note} |")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--gens", type=int, default=200)
    ap.add_argument("--teacher-games", type=int, default=1000)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--mix-games", type=int, default=150)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--window", type=int, default=200_000)
    ap.add_argument("--reuse", type=float, default=6.0)
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--stop-games", type=int, default=60)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--start-version", type=int, default=2)
    ap.add_argument("--qmix", type=float, default=0.0)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--load-data", action="store_true")
    ap.add_argument("--eval-first", action="store_true")
    ap.add_argument("--best-rate", type=float, default=0.0)
    ap.add_argument("--best", type=str, default="")
    ap.add_argument("--full-every", type=int, default=3)
    ap.add_argument("--boot-steps", type=int, default=1500)
    args = ap.parse_args()

    CKPT.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    dev = "cuda"
    torch.manual_seed(0)
    net = Net(args.channels, args.blocks).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    buf = Buffer(args.window)
    best = None
    ver = args.start_version
    if args.resume:
        st = torch.load(args.resume, map_location=dev)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        best = st.get("best")
        if args.best:
            best = args.best
        if isinstance(best, str):
            best = (best, args.best_rate)
        log("resumed", args.resume, "best", best)
    if args.load_data:
        import pickle
        for f in sorted(DATA.glob("games_*.pkl"), key=lambda q: q.stat().st_mtime):
            with open(f, "rb") as fh:
                for g in pickle.load(fh):
                    buf.add(g)
        buf.trim()
        log("loaded data, buffer", buf.n())

    pool = get_context("spawn").Pool(args.workers)
    seed = int(time.time()) % 100000 * 1000
    if args.eval_first:
        best = evaluate(MODELS / f"az_v{ver - 1}.onnx", None, args, ver - 1)

    if args.teacher_games > 0:
        t = time.time()
        games = run_jobs(pool, sp_jobs(STOPGAP, STOPGAP, args.teacher_games, args.sims, seed, 12, 0.3),
                         f"teacher_{seed}")
        seed += 10000
        for g in games:
            buf.add(g)
        buf.trim()
        log(f"teacher stopgap-vs-stopgap: {summarize(games)} {time.time() - t:.0f}s buf {buf.n()}")
        lp, lv, n = train(net, opt, buf, args.boot_steps, device=dev, qmix=args.qmix)
        log(f"boot train: lpol {lp:.3f} lval {lv:.3f} N {n}")
        cur = export(net, MODELS / f"az_v{ver}.onnx")
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "best": best,
                    "cfg": dict(c=args.channels, b=args.blocks)}, CKPT / f"az_v{ver}.pt")
        best = evaluate(cur, best, args, ver)
        ver += 1
    else:
        cur = MODELS / f"az_v{ver - 1}.onnx"

    for gen in range(args.gens):
        t = time.time()
        jobs = sp_jobs(cur, cur, args.games, args.sims, seed, 12, 0.3)
        jobs += sp_jobs(cur, STOPGAP, args.mix_games, args.sims, seed + args.games, 12, 0.3, alternate=True)
        games = run_jobs(pool, jobs, f"v{ver}_{seed}")
        seed += 10000
        for g in games:
            buf.add(g)
        buf.trim()
        new = sum(g["n"] for g in games)
        log(f"gen v{ver}: selfplay+mix {summarize(games)} {time.time() - t:.0f}s buf {buf.n()}")
        steps = int(args.reuse * new / 512)
        lp, lv, n = train(net, opt, buf, steps, device=dev, qmix=args.qmix)
        log(f"gen v{ver}: train {steps} steps lpol {lp:.3f} lval {lv:.3f}")
        cur = export(net, MODELS / f"az_v{ver}.onnx")
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "best": best,
                    "cfg": dict(c=args.channels, b=args.blocks)}, CKPT / f"az_v{ver}.pt")
        best = evaluate(cur, best, args, ver, full=(gen % args.full_every == args.full_every - 1))
        ver += 1


