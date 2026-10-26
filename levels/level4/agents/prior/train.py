"""Train the policy delta on deep-search visit targets (GPU), hold out games, export ONNX.

    uv run python levels/level4/agents/prior/train.py <out_name> [--tsharp 1.0] [--epochs 40] [--glob gen_*.pkl]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, INV, INVIDX, load_games, games_to_arrays, load_fast, PriorNet  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--tsharp", type=float, default=1.0)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--glob", default="gen_*.pkl")
ap.add_argument("--ch", type=int, default=32)
ap.add_argument("--blocks", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--warm_w", type=float, default=2.0, help="loss weight for warmup moves (<=8)")
ap.add_argument("--no_sym", action="store_true")
ap.add_argument("--all", action="store_true", help="train on all games (no holdout)")
ap.add_argument("--export", action="store_true")
ap.add_argument("--base", default="hand")
ap.add_argument("--qgap", type=float, default=0.0, help="if >0: weight = clip(gap/qgap, 0.25, 3)")
args = ap.parse_args()

torch.manual_seed(0)
dev = "cuda"
files = sorted((HERE / "data").glob(args.glob))
games = load_games(files)
key = np.array([(hash((g["seed"], g["game"])) % 10) for g in games])
# deterministic split: game index mod 10 == 0 -> validation
val_g = [g for g in games if (g["seed"] * 7919 + g["game"]) % 10 == 0]
tr_g = games if args.all else [g for g in games if (g["seed"] * 7919 + g["game"]) % 10 != 0]
Xtr, Ttr, Mtr, MVtr, Btr = games_to_arrays(tr_g, args.tsharp)
Xva, Tva, Mva, MVva, Bva = games_to_arrays(val_g, 1.0)
print(f"games {len(games)} train pos {len(Xtr)} val pos {len(Xva)} ({len(val_g)} games)", flush=True)

if args.base == "az":
    from common_az import load_az, AzPrior
    model = AzPrior(load_az(), ch=args.ch, blocks=args.blocks).to(dev)
elif args.base == "4f":
    from common_4f import load_4f, Prior4f
    model = Prior4f(load_4f(), ch=args.ch, blocks=args.blocks).to(dev)
else:
    model = PriorNet(load_fast(), ch=args.ch, blocks=args.blocks).to(dev)


def qgaps(gs):
    out = []
    for g in gs:
        for r in g["recs"]:
            if len(r["idx"]) == 0 or r["cnt"].sum() <= 0:
                continue
            o = np.argsort(-r["cnt"].astype(np.int64), kind="stable")
            out.append(abs(float(r["q"][o[0]] - r["q"][o[1]])) if len(o) > 1 else 1.0)
    return np.array(out, np.float32)


Wtr = np.where(MVtr <= 8, args.warm_w, 1.0).astype(np.float32)
if args.qgap > 0:
    gp = qgaps(tr_g)
    Wtr = Wtr * np.clip(gp / args.qgap, 0.25, 3.0)
    print(f"qgap weights: median gap {np.median(gp):.3f}, mean w {Wtr.mean():.2f}")
Xtr_t = torch.from_numpy(Xtr).to(dev).reshape(-1, 16, 64)
Ttr_t = torch.from_numpy(Ttr).to(dev)
Mtr_t = torch.from_numpy(Mtr).to(dev)
Wtr_t = torch.from_numpy(Wtr).to(dev)
Xva_t = torch.from_numpy(Xva).to(dev)
Tva_t = torch.from_numpy(Tva).to(dev)
Mva_t = torch.from_numpy(Mva).to(dev)
INV_t = torch.from_numpy(INV).to(dev)
INVIDX_t = torch.from_numpy(INVIDX).to(dev)


def masked_ce(logits, t, m):
    lp = torch.log_softmax(logits.masked_fill(~m, -1e4), 1)
    return -(t * lp).sum(1)


@torch.no_grad()
def evaluate():
    model.eval()
    ce, top = [], []
    for i in range(0, len(Xva_t), 1024):
        lg, _ = model(Xva_t[i:i + 1024])
        ce.append(masked_ce(lg, Tva_t[i:i + 1024], Mva_t[i:i + 1024]))
        top.append(lg.masked_fill(~Mva_t[i:i + 1024], -1e9).argmax(1))
    ce = torch.cat(ce).cpu().numpy(); top = torch.cat(top).cpu().numpy()
    ag = top == Bva
    w = MVva <= 8
    model.train()
    return dict(ce=float(ce.mean()), agree=float(ag.mean()), agree_warm=float(ag[w].mean()),
                agree_post=float(ag[~w].mean()), ce_warm=float(ce[w].mean()))


r0 = evaluate()
print("step0 (= hand_v3):", json.dumps(r0), flush=True)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                        weight_decay=args.wd)
bs = 256
steps_per = (len(Xtr_t) + bs - 1) // bs
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per,
                                            pct_start=0.15)
best = (1e9, None, None)
t0 = time.time()
for ep in range(args.epochs):
    perm = torch.randperm(len(Xtr_t), device=dev)
    tl = 0.0
    for i in range(0, len(perm), bs):
        b = perm[i:i + bs]
        x = Xtr_t[b]; t = Ttr_t[b]; m = Mtr_t[b]
        if not args.no_sym:
            s = torch.randint(0, 8, (len(b),), device=dev)
            x = torch.gather(x, 2, INV_t[s][:, None, :].expand(-1, 16, -1))
            t = torch.gather(t, 1, INVIDX_t[s]); m = torch.gather(m, 1, INVIDX_t[s])
        lg, _ = model(x.reshape(-1, 16, 8, 8))
        loss = (masked_ce(lg, t, m) * Wtr_t[b]).sum() / Wtr_t[b].sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tl += loss.item() * len(b)
    r = evaluate()
    if r["ce"] < best[0]:
        best = (r["ce"], {k: v.detach().clone() for k, v in model.state_dict().items()}, dict(r, epoch=ep))
    print(f"ep {ep} train {tl / len(Xtr_t):.4f} val {json.dumps(r)} T={float(torch.exp(model.log_t)):.2f} "
          f"{time.time() - t0:.0f}s", flush=True)

final = dict(r, epoch=args.epochs - 1)
use = best if not args.all else (None, {k: v.detach().clone() for k, v in model.state_dict().items()}, final)
model.load_state_dict(use[1])
print("best:", json.dumps(use[2]))
torch.save(dict(state=use[1], args=vars(args), step0=r0, best=use[2]), HERE / f"{args.name}.pt")
with open(HERE / "train_log.jsonl", "a") as f:
    f.write(json.dumps(dict(name=args.name, args=vars(args), step0=r0, best=use[2],
                            ntrain=len(Xtr), nval=len(Xva))) + "\n")
if args.export:
    from c1 import export_onnx, MODELS
    out = MODELS / f"{args.name}.onnx"
    for k in range(4):
        try:
            export_onnx(model.cpu(), out)
            print("exported + validated", out)
            break
        except RuntimeError as e:
            print("validate failed, retry", str(e)[-300:])
