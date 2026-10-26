"""SPRINT-2 fine-tune from az_res4f_v6 on fresh v6 self-play (x2) + az buffer + live + teacher/oracle.
uv run python levels/level4/agents/sprint2/ft.py OUT_NAME STEPS [batch] [lr]"""
if __name__ == "__main__":
    import copy, math, os, pickle, sys, time
    from pathlib import Path
    HERE = Path(__file__).resolve().parent
    L4 = HERE.parents[1]
    AZ = L4 / "agents" / "az"
    sys.path.insert(0, str(AZ)); sys.path.insert(0, str(L4)); sys.path.insert(0, str(L4 / "reviews"))
    import numpy as np
    import torch
    import torch.nn.functional as F
    import reslib
    from azlib import D4_IDX, D4_INV
    from reslib import DATA, MODELS, RCKPT, export, graph_size, load_hand_sp, validate

    def log(*a):
        print(time.strftime("%H:%M:%S"), *a, flush=True)

    name, steps = sys.argv[1], int(sys.argv[2])
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    lr_max = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
    PWS = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    torch.manual_seed(7)
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.3)
    parts = []
    old = ("games_res_v4_", "games_res_v5_", "games_res_v6_", "games_res_v7_", "games_res_v8_")
    for f in sorted(DATA.glob("games_res_*.pkl")):
        g = pickle.load(open(f, "rb"))
        parts.append((g, 0.5 if f.name.startswith(old) else 1.0, "sp"))
    for f in sorted(DATA.glob("games_end_train[23]*.pkl")):
        parts.append((pickle.load(open(f, "rb")), 1.0, "end"))
    hd = L4 / "agents" / "hand" / "data"
    parts.append(([load_hand_sp(hd / "sp_v4f.pkl")], 1.0, "sp"))
    dd = L4 / "agents" / "distill" / "data"
    parts.append(([load_hand_sp(dd / "endgame_labels.pkl")], 2.0, "teacher"))
    parts.append(([load_hand_sp(dd / "endgame_live.pkl")], 3.0, "teacher"))
    og = []
    for f in sorted(dd.glob("oracle800_az_res4f_v2_*.npz")):
        d = np.load(f, allow_pickle=True)
        for i in range(len(d["obs"])):
            li = np.nonzero(d["legal"][i])[0].astype(np.int16)
            vis = d["visits"][i][li].astype(np.float32)
            if vis.sum() <= 0:
                continue
            og.append(dict(obs=np.rint(d["obs"][i:i + 1] * 8).astype(np.uint8), idx=[li], p=[vis / vis.sum()],
                           z=np.array([d["root_q"][i]], np.float32), q=np.array([d["root_q"][i]], np.float32), n=1))
    parts.append((og, 3.0, "oracle"))
    lv = L4 / "agents" / "live"
    for fn, imitate in (("live_opp.pkl", True), ("live_ours.pkl", False)):
        g = load_hand_sp(lv / fn)
        g["p"] = [pp_ if (imitate and zz > 0) else np.zeros_like(pp_) for pp_, zz in zip(g["p"], g["z"])]
        g["q"] = g["z"].copy()
        parts.append(([g], 2.0, "live"))
    import os as _os
    for f in [x for x in _os.environ.get("S2_EXTRA", "").split(";") if x]:
        parts.append(([load_hand_sp(Path(f))], 1.0, "sp"))
        log(f"extra {f}")
    fresh = []
    for f in sorted((HERE / "data").glob("sp_*.pkl")):
        fresh += pickle.load(open(f, "rb"))
    parts.append((fresh, 2.0, "fresh"))
    log(f"fresh games {len(fresh)} pos {sum(g['n'] for g in fresh)}")

    obs_l, idx_l, p_l, zt_l, w_l, pw_l = [], [], [], [], [], []
    for games, w, kind in parts:
        for g in games:
            o = g["obs"]
            obs_l.append(o)
            idx_l += list(g["idx"]); p_l += list(g["p"])
            z = g["z"].astype(np.float32); q = g["q"].astype(np.float32)
            own = o[:, 0:5].reshape(len(o), -1).sum(1) / 8.0
            en = o[:, 6:11].reshape(len(o), -1).sum(1) / 8.0
            low = (own <= 2) & (en <= 2)
            zt = z if kind in ("teacher", "oracle", "live") else np.where(low, z, 0.7 * z + 0.3 * q)
            zt_l.append(zt.astype(np.float32))
            mag = o[:, 15, 0, 0] > 0
            ww = np.full(len(o), w, np.float32) * np.where(mag, 2.0, 1.0)
            ww *= np.where((own <= 1) & (en <= 1), 1.5, 1.0)
            w_l.append(ww)
            pw_l.append(np.full(len(o), PWS if kind in ("sp", "fresh") else 1.0, np.float32))
    obs = np.concatenate(obs_l); zt = np.concatenate(zt_l); wt = np.concatenate(w_l); pwt = np.concatenate(pw_l)
    L = max(len(a) for a in idx_l)
    idx = np.full((len(idx_l), L), -1, np.int16); pp = np.zeros((len(idx_l), L), np.float16)
    for i, (a, b) in enumerate(zip(idx_l, p_l)):
        idx[i, :len(a)] = a; pp[i, :len(b)] = b
    counts = {k: sum(sum(g["n"] for g in gs) for gs, _, kk in parts if kk == k) for k in ("sp", "end", "teacher", "oracle", "live", "fresh")}
    log(f"N {len(obs)} L {L} {counts} magenta {np.mean(obs[:, 15, 0, 0] > 0):.2f}")
    O = torch.from_numpy(obs).to(dev); I = torch.from_numpy(idx).to(dev); P = torch.from_numpy(pp).to(dev)
    Z = torch.from_numpy(zt).to(dev); W = torch.from_numpy(wt).to(dev); PW = torch.from_numpy(pwt).to(dev)
    inv = torch.from_numpy(D4_INV).to(dev); imap = torch.from_numpy(D4_IDX).to(dev)

    net = reslib.ResFast(32, 2, kind="v4f").to(dev)
    net.load_state_dict(torch.load(os.environ.get("S2_INIT", str(RCKPT / "az_res4f_v6.pt")), map_location=dev)["net"])
    params = [q for q in net.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr_max, weight_decay=1e-4)

    def step_fn():
        b = torch.multinomial(W, B, replacement=True)
        x = O[b].float() / 8.0
        bi = I[b].long(); pad = bi < 0; bi = bi.clamp(min=0)
        sy = torch.randint(0, 8, (B,), device=dev)
        x = x.reshape(B, 16, 64).gather(2, inv[sy][:, None, :].expand(B, 16, 64)).reshape(B, 16, 8, 8)
        bi = imap[sy].gather(1, bi)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, v = net(x)
        logits = logits.float(); v = v.float()
        lg_ = logits.gather(1, bi).masked_fill(pad, -1e9)
        logp = F.log_softmax(lg_, 1).masked_fill(pad, 0.0)
        lpol = -(PW[b] * (P[b].float() * logp).sum(1)).sum() / PW[b].sum()
        lval = F.mse_loss(v, Z[b])
        loss = lpol + lval
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return lpol.item(), lval.item()

    net.train()
    t0 = time.time()
    swa_from = int(steps * 0.8)
    avg = None; n_avg = 0
    for s in range(steps):
        lr = lr_max * (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
        lr = max(lr, lr_max * 0.1) if s >= swa_from else lr
        for g_ in opt.param_groups:
            g_["lr"] = lr if s > steps * 0.03 else lr_max * (s + 1) / (steps * 0.03)
        a_, b_ = step_fn()
        if s % 500 == 0:
            log(f"step {s} lpol {a_:.3f} lval {b_:.3f} lr {lr:.2e} {time.time()-t0:.0f}s")
        if s >= swa_from and (s - swa_from) % 50 == 0:
            if avg is None:
                avg = copy.deepcopy(net); n_avg = 1
            else:
                n_avg += 1
                with torch.no_grad():
                    for pa, pn in zip(avg.parameters(), net.parameters()):
                        if pn.requires_grad:
                            pa.mul_((n_avg - 1) / n_avg).add_(pn / n_avg)
    log(f"trained {steps} steps in {time.time() - t0:.0f}s SWA {n_avg}, peak mem {torch.cuda.max_memory_allocated()/2**30:.2f} GB")
    for m_ in avg.modules():
        if isinstance(m_, torch.nn.modules.batchnorm._BatchNorm):
            m_.reset_running_stats(); m_.momentum = None
    avg.train()
    with torch.no_grad():
        sel = torch.randint(0, len(O), (8192,), device=dev)
        for i_ in range(0, 8192, 1024):
            avg(O[sel[i_:i_ + 1024]].float() / 8.0)
    avg.eval()
    (HERE / "ckpt").mkdir(exist_ok=True)
    torch.save({"net": avg.state_dict()}, HERE / "ckpt" / f"{name}.pt")
    out = HERE / f"{name}.onnx"
    export(avg, out)
    log(f"{out.name} graph {graph_size(out)}")
    ok, msg = validate(out)
    log(f"validator {'ACCEPTED' if ok else 'REJECTED'} {msg if not ok else ''}")
