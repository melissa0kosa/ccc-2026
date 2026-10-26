"""SPRINT-3: fine-tune az_res4f_v6 prior towards sharpened 800-sim visit distributions (policy ExIt).

uv run python levels/level4/agents/sprint3/train.py OUT.onnx STEPS LR [polw]
env: S3_POW (target sharpening, default 4), S3_MISW (weight where 800-best != v6 top-1, default 3),
     S3_MODE distill (value distilled to frozen v6 value; default) | freeze (train policy heads only, value exact)
"""
if __name__ == "__main__":
    import copy
    import glob
    import math
    import os
    import pickle
    import sys
    import time
    from pathlib import Path

    HERE = Path(__file__).resolve().parent
    L4 = HERE.parents[1]
    AZ = L4 / "agents" / "az"
    sys.path.insert(0, str(AZ)); sys.path.insert(0, str(L4))
    import numpy as np
    import torch
    import torch.nn.functional as F

    import reslib
    from azlib import D4_IDX, D4_INV
    from reslib import DATA, MODELS, RCKPT, export, graph_size, load_hand_sp, log

    out_path = Path(sys.argv[1]); steps = int(sys.argv[2]); lr_max = float(sys.argv[3])
    polw = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
    POW = float(os.environ.get("S3_POW", "4"))
    MISW = float(os.environ.get("S3_MISW", "3"))
    MODE = os.environ.get("S3_MODE", "distill")
    torch.manual_seed(int(os.environ.get("S3_SEED", "0")))
    dev = "cuda"

    def read_stream(fn):
        out = []
        with open(fn, "rb") as f:
            while True:
                try:
                    out.append(pickle.load(f))
                except Exception:  # noqa: BLE001 (partial file being appended)
                    break
        return out

    # ------------------------------------------------------------- labels
    lab, held = [], []  # (obs u8, idx, p_sharp, w, best)
    hold_files = os.environ.get("S3_HOLDFILES", "lab_31008.pkl").split(",")

    def mk(obs_u8, idx, cnt, w):
        c = np.asarray(cnt, np.float64)
        if c.sum() <= 0:
            return None
        p = c ** POW
        p = (p / p.sum()).astype(np.float32)
        return (obs_u8, np.asarray(idx, np.int64), p, w, int(np.argmax(c)))

    for fn in sorted(glob.glob(str(HERE / "data" / "lab_*.pkl"))):
        is_h = Path(fn).name in hold_files
        for g in read_stream(fn):
            for r in g["recs"]:
                t = mk(r["obs"], r["idx"], r["cnt"], 1.0)
                if t is not None:
                    (held if is_h else lab).append(t)
    n_new = len(lab)
    pw = float(os.environ.get("S3_PRIORW", "0.5"))
    if pw > 0:
        for fn in sorted(glob.glob(str(L4 / "agents" / "prior" / "data" / "gen_r4f_*.pkl"))):
            for g in read_stream(fn):
                for r in g["recs"]:
                    t = mk(np.rint(r["obs"] * 8).astype(np.uint8), r["idx"], r["cnt"], pw)
                    if t is not None:
                        lab.append(t)
    log(f"s3: labels new {n_new} (+{len(lab) - n_new} prior v2-labelled w{pw}), held-out {len(held)} from {hold_files}")

    def pack(recs):
        obs = np.stack([r[0] for r in recs])
        L = max(len(r[1]) for r in recs)
        idx = np.full((len(recs), L), -1, np.int64); p = np.zeros((len(recs), L), np.float32)
        for i, r in enumerate(recs):
            idx[i, :len(r[1])] = r[1]; p[i, :len(r[2])] = r[2]
        return (torch.from_numpy(obs).to(dev), torch.from_numpy(idx).to(dev), torch.from_numpy(p).to(dev),
                torch.tensor([r[3] for r in recs], dtype=torch.float32, device=dev),
                torch.tensor([r[4] for r in recs], dtype=torch.int64, device=dev))

    LO, LI, LP, LW, LB = pack(lab)
    HO, HI, HP, _, HB = pack(held)

    # ------------------------------------------------------------- replay
    parts = []
    for fn in ("games_res_4f2x.pkl", "games_end_train3.pkl"):
        parts.append(pickle.load(open(DATA / fn, "rb")))
    parts.append([load_hand_sp(L4 / "agents" / "hand" / "data" / "sp_v4f.pkl")])
    ro, ri, rp = [], [], []
    for games in parts:
        for g in games:
            ro.append(g["obs"]); ri += list(g["idx"]); rp += list(g["p"])
    ro = np.concatenate(ro)
    L = max(len(a) for a in ri)
    rI = np.full((len(ri), L), -1, np.int64); rP = np.zeros((len(ri), L), np.float32)
    for i, (a, b) in enumerate(zip(ri, rp)):
        rI[i, :len(a)] = a; rP[i, :len(b)] = b
    RO = torch.from_numpy(ro).to(dev); RI = torch.from_numpy(rI).to(dev); RP = torch.from_numpy(rP).to(dev)
    Wd = max(LI.shape[1], RI.shape[1])

    def padw(I_, P_):
        k = Wd - I_.shape[1]
        if k > 0:
            I_ = torch.cat([I_, torch.full((len(I_), k), -1, dtype=I_.dtype, device=dev)], 1)
            P_ = torch.cat([P_, torch.zeros((len(P_), k), dtype=P_.dtype, device=dev)], 1)
        return I_, P_

    LI, LP = padw(LI, LP); RI, RP = padw(RI, RP)
    log(f"s3: replay {len(RO)} positions")
    inv = torch.from_numpy(D4_INV).to(dev); imap = torch.from_numpy(D4_IDX).to(dev)

    net = reslib.ResFast(32, 2, kind="v4f").to(dev)
    net.load_state_dict(torch.load(os.environ.get("S3_INIT", str(RCKPT / "az_res4f_v6.pt")), map_location=dev)["net"])
    teacher = copy.deepcopy(net).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)

    def legal_logits(m, O, I):
        logits, v = m(O.float() / 8.0)
        pad = I < 0
        return logits.gather(1, I.clamp(min=0)).masked_fill(pad, -1e9), v

    # mismatch weights: 800-sim best != v6 prior top-1
    with torch.no_grad():
        top = torch.cat([legal_logits(teacher, LO[i:i + 2048], LI[i:i + 2048])[0].argmax(1)
                         for i in range(0, len(LO), 2048)])
    mis = (top != LB).float()
    LW = LW * (1 + (MISW - 1) * mis)
    log(f"s3: train mismatch frac {float(mis.mean()):.3f} (weight x{MISW}), target pow {POW}, mode {MODE}")

    if MODE == "freeze":
        for n_, q in net.named_parameters():
            q.requires_grad_(n_.startswith(("pf.", "pt.", "p64.")))
    params = [q for q in net.parameters() if q.requires_grad]
    log(f"s3: trainable params {sum(q.numel() for q in params)}")
    opt = torch.optim.AdamW(params, lr=lr_max, weight_decay=1e-4)

    def held_eval(m):
        m.eval()
        with torch.no_grad():
            lg_, v = legal_logits(m, HO, HI)
            _, vt = teacher(HO.float() / 8.0)
            logp = F.log_softmax(lg_, 1).masked_fill(HI < 0, 0.0)
            ce = float(-(HP * logp).sum(1).mean())
            hit = lg_.argmax(1) == HB
            top1 = float(hit.float().mean())
            dv = float((v - vt).abs().max())
        return ce, top1, dv

    # parity check vs deployed v6 onnx
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    s6 = ort.InferenceSession(os.environ.get("S3_REF", str(MODELS / "az_res4f_v6.onnx")), so, providers=["CPUExecutionProvider"])
    x = (HO[:64].float() / 8.0).cpu().numpy()
    a = s6.run(None, {"observation": x})
    with torch.no_grad():
        b = teacher(torch.from_numpy(x).to(dev))
    log(f"s3: ckpt vs v6.onnx parity max|dp| {np.abs(a[0] - b[0].cpu().numpy()).max():.2e} "
        f"max|dv| {np.abs(a[1] - b[1].cpu().numpy()).max():.2e}")
    ce0, t0_, _ = held_eval(net)
    log(f"s3: BEFORE held-out CE {ce0:.3f} top1 {t0_:.3f}")

    B = int(os.environ.get("S3_B", "512"))  # per half

    def step_fn():
        bl = torch.multinomial(LW, B, replacement=True)
        br = torch.randint(0, len(RO), (B,), device=dev)
        O = torch.cat([LO[bl], RO[br]]); I = torch.cat([LI[bl], RI[br]]); P = torch.cat([LP[bl], RP[br]])
        n = 2 * B
        x = O.float() / 8.0
        pad = I < 0; bi = I.clamp(min=0)
        sy = torch.randint(0, 8, (n,), device=dev)
        x = x.reshape(n, 16, 64).gather(2, inv[sy][:, None, :].expand(n, 16, 64)).reshape(n, 16, 8, 8)
        bi = imap[sy].gather(1, bi)
        with torch.no_grad():
            _, vt = teacher(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, v = net(x)
        logits = logits.float(); v = v.float()
        lg_ = logits.gather(1, bi).masked_fill(pad, -1e9)
        logp = F.log_softmax(lg_, 1).masked_fill(pad, 0.0)
        lp_each = -(P * logp).sum(1)
        lpl, lpr = lp_each[:B].mean(), lp_each[B:].mean()
        lv = F.mse_loss(v, vt)
        loss = polw * lpl + lpr + (0.0 if MODE == "freeze" else 4.0 * lv)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return lpl.item(), lpr.item(), lv.item()

    def set_mode():
        if MODE == "freeze":
            net.eval()  # keep trunk BN stats fixed -> value identical
        else:
            net.train()

    set_mode()
    swa_from = int(steps * 0.8)
    avg = None; n_avg = 0
    t0 = time.time()
    for s in range(steps):
        lr = lr_max * 0.5 * (1 + math.cos(math.pi * s / steps))
        if s >= swa_from:
            lr = max(lr, lr_max * 0.1)
        if s < steps * 0.03:
            lr = lr_max * (s + 1) / (steps * 0.03)
        for g_ in opt.param_groups:
            g_["lr"] = lr
        a_ = step_fn()
        if s % 200 == 0 or s == steps - 1:
            ce, t1, dv = held_eval(net); set_mode()
            log(f"s3 step {s}: lpol_lab {a_[0]:.3f} lpol_rep {a_[1]:.3f} lval(distill) {a_[2]:.4f} | "
                f"held CE {ce:.3f} top1 {t1:.3f} max|dv| {dv:.3f}")
        if s >= swa_from and (s - swa_from) % 25 == 0:
            if avg is None:
                avg = copy.deepcopy(net); n_avg = 1
            else:
                n_avg += 1
                with torch.no_grad():
                    for pa, pn in zip(avg.parameters(), net.parameters()):
                        if pn.requires_grad:
                            pa.mul_((n_avg - 1) / n_avg).add_(pn / n_avg)
    log(f"s3: trained {steps} steps in {time.time() - t0:.0f}s, SWA {n_avg}")
    if MODE != "freeze":
        for m_ in avg.modules():
            if isinstance(m_, torch.nn.modules.batchnorm._BatchNorm):
                m_.reset_running_stats(); m_.momentum = None
        avg.train()
        with torch.no_grad():
            for _ in range(8):
                bl = torch.randint(0, len(LO), (512,), device=dev); br = torch.randint(0, len(RO), (512,), device=dev)
                avg(torch.cat([LO[bl], RO[br]]).float() / 8.0)
    avg.eval()
    ce1, t1, dv1 = held_eval(avg)
    log(f"s3: AFTER (SWA) held-out CE {ce0:.3f}->{ce1:.3f} top1(800-best) {t0_:.3f}->{t1:.3f} max|dv| vs v6 {dv1:.3f}")
    export(avg, out_path)
    torch.save({"net": avg.state_dict()}, HERE / (out_path.stem + ".pt"))
    log(f"s3: exported {out_path} graph {graph_size(out_path)}")
