"""GPU sprint: big-batch bf16 retrain of the hand_v4f residual from az_res4f_v4 on ALL data, weighted sampling.

uv run python levels/level4/agents/az/sprint.py OUT_VERSION SECONDS [init_ckpt]
"""
if __name__ == "__main__":
    import copy
    import json
    import math
    import os
    import pickle
    import sys
    import time
    from pathlib import Path

    HERE = Path(__file__).resolve().parent
    L4 = HERE.parents[1]
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(L4)); sys.path.insert(0, str(L4 / "reviews"))
    import numpy as np
    import torch
    import torch.nn.functional as F

    import reslib
    from azlib import D4_IDX, D4_INV
    from reslib import DATA, MODELS, RCKPT, export, graph_size, load_hand_sp, log, validate

    ver, budget = int(sys.argv[1]), float(sys.argv[2])
    init = sys.argv[3] if len(sys.argv) > 3 else str(RCKPT / "az_res4f_v4.pt")
    torch.manual_seed(ver)
    dev = "cuda"

    # ------------------------------------------------------------------ data
    parts = []  # (games list, weight multiplier, kind)

    def add_games(games, w, kind):
        parts.append((games, w, kind))

    for f in sorted(DATA.glob("games_res_*.pkl")):
        add_games(pickle.load(open(f, "rb")), 1.0, "sp")
    for f in sorted(DATA.glob("games_end_train[23]*.pkl")):
        add_games(pickle.load(open(f, "rb")), 1.0, "end")
    hd = L4 / "agents" / "hand" / "data"
    add_games([load_hand_sp(hd / "sp_v4f.pkl")], 1.0, "sp")
    dd = L4 / "agents" / "distill" / "data"
    add_games([load_hand_sp(dd / "endgame_labels.pkl")], 2.0, "teacher")
    add_games([load_hand_sp(dd / "endgame_live.pkl")], 3.0, "teacher")
    # oracle 800-sim labels: policy = visits, value target = 800-sim root Q
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
    add_games(og, 3.0, "oracle")
    # live games: outcome-only value samples (policy target empty)
    from analyze_matches import parse
    from c1 import ColonyOneEnv
    seen, lg = set(), []
    for f in sorted((L4 / "data" / "matches").glob("*.json")):
        try:
            ms = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(ms, list):
            continue
        for m in ms:
            if not isinstance(m, dict) or not m.get("protocol") or m.get("match_id") in seen:
                continue
            seen.add(m["match_id"])
            try:
                mv, _ = parse(m["protocol"])
                win = "CYAN" if "CYAN WINS" in m["protocol"] else "MAGENTA"
                env = ColonyOneEnv()
                obs, idx, p, z = [], [], [], []
                for team, _pc, a, b, _k in mv:
                    if env.terminated:
                        break
                    li = env.legal_action_indices().astype(np.int16)
                    obs.append(np.rint(env.observation() * 8).astype(np.uint8))
                    idx.append(li); p.append(np.zeros(len(li), np.float32))
                    z.append(1.0 if env.current_team == win else -1.0)
                    c = [x for x in env.legal_actions() if env.format_action(x).replace(" ", "") == f"{a}->{b}"]
                    if len(c) != 1:
                        break
                    env.step(c[0])
                if obs:
                    zz = np.array(z, np.float32)
                    lg.append(dict(obs=np.stack(obs), idx=idx, p=p, z=zz, q=zz.copy(), n=len(obs)))
            except Exception:  # noqa: BLE001
                continue
    lv_dir = L4 / "agents" / "live"
    if os.environ.get("AZ_LIVEREC", "0") == "1" and (lv_dir / "live_opp.pkl").exists():
        wl = float(os.environ.get("AZ_LIVEREC_W", "5"))
        for fn, imitate in (("live_opp.pkl", True), ("live_ours.pkl", False)):
            g = load_hand_sp(lv_dir / fn)
            # policy: imitate only opponents' moves from games they went on to win; otherwise value-only
            g["p"] = [pp_ if (imitate and zz > 0) else np.zeros_like(pp_) for pp_, zz in zip(g["p"], g["z"])]
            g["q"] = g["z"].copy()
            add_games([g], wl, "live")
        log(f"sprint: live records added (weight {wl}); replayed-protocol live games skipped (same games)")
    else:
        add_games(lg, 3.0, "live")

    obs_l, idx_l, p_l, zt_l, w_l, pol_l = [], [], [], [], [], []
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
            ww = np.full(len(o), w, np.float32) * np.where(mag, 2.0, 1.0)  # MAGENTA-to-move x2
            ww *= np.where((own <= 1) & (en <= 1), 1.5, 1.0)
            w_l.append(ww)
    obs = np.concatenate(obs_l); zt = np.concatenate(zt_l); wt = np.concatenate(w_l)
    L = max(len(a) for a in idx_l)
    idx = np.full((len(idx_l), L), -1, np.int64); pp = np.zeros((len(idx_l), L), np.float32)
    for i, (a, b) in enumerate(zip(idx_l, p_l)):
        idx[i, :len(a)] = a; pp[i, :len(b)] = b
    counts = {k: sum(sum(g["n"] for g in gs) for gs, _, kk in parts if kk == k) for k in ("sp", "end", "teacher", "oracle", "live")}
    log(f"sprint v{ver}: N {len(obs)} {counts} live games {len(lg)} magenta {np.mean(obs[:, 15, 0, 0] > 0):.2f}")
    O = torch.from_numpy(obs).to(dev); I = torch.from_numpy(idx).to(dev); P = torch.from_numpy(pp).to(dev)
    Z = torch.from_numpy(zt).to(dev); W = torch.from_numpy(wt).to(dev)
    inv = torch.from_numpy(D4_INV).to(dev); imap = torch.from_numpy(D4_IDX).to(dev)

    # ------------------------------------------------------------------ model
    net = reslib.ResFast(32, 2, kind="v4f").to(dev)
    net.load_state_dict(torch.load(init, map_location=dev)["net"])
    params = [q for q in net.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    B = int(os.environ.get("AZ_B", "2048"))
    lr_max = float(os.environ.get("AZ_LR", "4e-4"))

    def step_fn():
        b = torch.multinomial(W, B, replacement=True)
        x = O[b].float() / 8.0
        bi = I[b]; pad = bi < 0; bi = bi.clamp(min=0)
        sy = torch.randint(0, 8, (B,), device=dev)
        x = x.reshape(B, 16, 64).gather(2, inv[sy][:, None, :].expand(B, 16, 64)).reshape(B, 16, 8, 8)
        bi = imap[sy].gather(1, bi)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, v = net(x)
        logits = logits.float(); v = v.float()
        lg_ = logits.gather(1, bi).masked_fill(pad, -1e9)
        logp = F.log_softmax(lg_, 1).masked_fill(pad, 0.0)
        lpol = -(P[b] * logp).sum(1).mean()
        lval = F.mse_loss(v, Z[b])
        loss = lpol + lval
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return lpol.item(), lval.item()

    net.train()
    t0 = time.time()
    for _ in range(20):
        for g_ in opt.param_groups:
            g_["lr"] = 1e-5
        step_fn()
    torch.cuda.synchronize()
    per = (time.time() - t0) / 20
    steps = int(os.environ["AZ_STEPS"]) if os.environ.get("AZ_STEPS") else max(500, int(budget / per))
    log(f"sprint v{ver}: {per * 1000:.0f} ms/step at batch {B} -> {steps} steps")
    swa_from = int(steps * 0.8)
    avg = None; n_avg = 0
    lp = lv = 0.0
    for s in range(steps):
        lr = lr_max * (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
        lr = max(lr, lr_max * 0.05) if s >= swa_from else lr
        for g_ in opt.param_groups:
            g_["lr"] = lr if s > steps * 0.03 else lr_max * s / (steps * 0.03)
        a_, b_ = step_fn()
        if s % 100 == 0:
            lp, lv = a_, b_
        if s >= swa_from and (s - swa_from) % 50 == 0:
            if avg is None:
                avg = copy.deepcopy(net); n_avg = 1
            else:
                n_avg += 1
                with torch.no_grad():
                    for pa, pn in zip(avg.parameters(), net.parameters()):
                        if pn.requires_grad:
                            pa.mul_((n_avg - 1) / n_avg).add_(pn / n_avg)
    log(f"sprint v{ver}: trained {steps} steps in {time.time() - t0:.0f}s, last lpol {lp:.3f} lval {lv:.3f}, SWA {n_avg}")
    for m_ in avg.modules():
        if isinstance(m_, torch.nn.modules.batchnorm._BatchNorm):
            m_.reset_running_stats(); m_.momentum = None
    avg.train()
    with torch.no_grad():
        sel = torch.randint(0, len(O), (8192,), device=dev)
        for i_ in range(0, 8192, 1024):
            avg(O[sel[i_:i_ + 1024]].float() / 8.0)
    avg.eval()
    out = MODELS / f"az_res4f_v{ver}.onnx"
    export(avg, out)
    torch.save({"net": avg.state_dict()}, RCKPT / f"az_res4f_v{ver}.pt")
    s125 = copy.deepcopy(avg)
    with torch.no_grad():
        s125.vfc2.weight.mul_(1.25); s125.vfc2.bias.mul_(1.25)
    out2 = MODELS / f"az_res4f_v{ver}s.onnx"
    export(s125, out2)
    for o_ in (out, out2):
        ok, _ = validate(o_)
        log(f"sprint: {o_.name} graph {graph_size(o_)} validator {'ACCEPTED' if ok else 'REJECTED'}")
