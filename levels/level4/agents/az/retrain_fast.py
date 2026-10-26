"""Offline retrain of the residual correction on the FastNet (hand_v2fast) base from saved self-play games.

uv run python levels/level4/agents/az/retrain_fast.py VERSION STEPS
Writes models/az_res_vVERSION.onnx and ckpt_res/az_res_vVERSION.pt (loop --resume format).
"""
if __name__ == "__main__":
    import pickle
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np
    import onnxruntime as ort
    import torch

    import reslib
    from reslib import DATA, HAND, HERE, MODELS, RCKPT, Buffer, export, graph_size, load_hand_sp, log, train, validate

    ver, steps = int(sys.argv[1]), int(sys.argv[2])
    qmix = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    kind = sys.argv[4] if len(sys.argv) > 4 else "v2"
    ref = MODELS / ("hand_v4f.onnx" if kind == "v4f" else "hand_v2.onnx")
    stem = "az_res4f" if kind == "v4f" else "az_res"
    torch.set_num_threads(2)
    torch.manual_seed(1)
    net = reslib.ResFast(32, 2, kind=kind).cuda()

    # step-0 parity vs hand_v2.onnx on real positions
    p0 = HERE / "exp" / f"{stem}_step0.onnx"
    p0.parent.mkdir(exist_ok=True)
    export(net, p0)
    games = pickle.load(open(sorted(DATA.glob("games_res_*.pkl"))[0], "rb"))
    x = np.concatenate([g["obs"] for g in games[:80]]).astype(np.float32) / 8.0
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    a = ort.InferenceSession(str(p0), so, providers=["CPUExecutionProvider"]).run(None, {"observation": x})
    b = ort.InferenceSession(str(ref), so, providers=["CPUExecutionProvider"]).run(None, {"observation": x})
    dp, dv = float(np.abs(a[0] - b[0]).max()), float(np.abs(a[1] - b[1]).max())
    log(f"resfast {stem} step0 parity vs {ref.stem} on {len(x)} positions: max|dpolicy| {dp:.2e} max|dvalue| {dv:.2e}; "
        f"graph {graph_size(p0)}")

    buf = Buffer(10**9)
    for f in sorted(DATA.glob("games_res_*.pkl"), key=lambda q: q.stat().st_mtime):
        for g in pickle.load(open(f, "rb")):
            buf.add(g)
    buf.add(load_hand_sp(HERE.parent / "hand" / "data" / "sp_v2.pkl"))
    if kind == "v4f" and (HERE.parent / "hand" / "data" / "sp_v4f.pkl").exists():
        buf.add(load_hand_sp(HERE.parent / "hand" / "data" / "sp_v4f.pkl"))
    end_frac = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    if end_frac > 0:
        eg = []
        import os as _os
        for f in sorted(DATA.glob(_os.environ.get("AZ_END_GLOB", "games_end_train*.pkl"))):
            eg += pickle.load(open(f, "rb"))
        n_end = sum(g["n"] for g in eg)
        rep = max(1, int(round(end_frac / (1 - end_frac) * buf.n() / max(1, n_end))))
        for _ in range(rep):
            for g in eg:
                buf.add(g)
        log(f"resfast: endgame data {len(eg)} games {n_end} positions x{rep} -> {n_end * rep / buf.n():.0%} of samples")
    if _os.environ.get("AZ_TEACHER"):
        tg = load_hand_sp(Path(_os.environ["AZ_TEACHER"]))
        trep = int(_os.environ.get("AZ_TEACHER_REP", "3"))
        for _ in range(trep):
            buf.add(tg)
        log(f"resfast: teacher endgame labels {tg['n']} positions x{trep}")
    if _os.environ.get("AZ_TEACHER2"):
        tg2 = load_hand_sp(Path(_os.environ["AZ_TEACHER2"]))
        for _ in range(int(_os.environ.get("AZ_TEACHER2_REP", "5"))):
            buf.add(tg2)
        log(f"resfast: teacher2 labels {tg2['n']} positions")
    log(f"resfast offline data: {buf.n()} positions")
    opt = torch.optim.AdamW([q for q in net.parameters() if q.requires_grad], lr=3e-4, weight_decay=1e-4)
    t = time.time()
    import os
    if os.environ.get("AZ_LOWMAT", "0") == "1":
        buf.lowmat_outcome = True
    sched = None
    if os.environ.get("AZ_COSINE", "0") == "1":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(os.environ.get("AZ_LR", "6e-4")),
                                                    total_steps=steps, pct_start=0.1)
    if os.environ.get("AZ_INIT"):
        net.load_state_dict(torch.load(os.environ["AZ_INIT"], map_location="cuda")["net"])
        log("resfast: init from", os.environ["AZ_INIT"])
    lp, lv, n = train(net, opt, buf, steps, qmix=qmix, sched=sched)
    log(f"resfast: lowmat outcome-only frac {getattr(buf, 'low_frac', 0):.2f}, sched {'onecycle' if sched else 'const'}")
    swa_steps = int(os.environ.get("AZ_SWA", "0"))
    if swa_steps > 0:
        # SWA within the run: continue at low constant LR, average weights every 50 steps, then recompute BN stats
        import copy as _copy
        for g_ in opt.param_groups:
            g_["lr"] = float(os.environ.get("AZ_SWA_LR", "1e-4"))
        avg = _copy.deepcopy(net)
        n_avg = 1
        for k_ in range(swa_steps // 50):
            train(net, opt, buf, 50, qmix=qmix)
            n_avg += 1
            with torch.no_grad():
                for pa, pn in zip(avg.parameters(), net.parameters()):
                    if pn.requires_grad:
                        pa.mul_((n_avg - 1) / n_avg).add_(pn / n_avg)
        # recompute BatchNorm running stats on ~4k training positions
        _gi = np.random.default_rng(0).choice(len(buf.games), size=min(400, len(buf.games)), replace=False)
        obs_all = torch.from_numpy(np.concatenate([buf.games[i]["obs"] for i in _gi])).cuda()
        for m_ in avg.modules():
            if isinstance(m_, torch.nn.modules.batchnorm._BatchNorm):
                m_.reset_running_stats()
                m_.momentum = None
        avg.train()
        with torch.no_grad():
            sel = torch.randperm(obs_all.shape[0], device="cuda")[:4096]
            for i_ in range(0, len(sel), 512):
                avg(obs_all[sel[i_:i_ + 512]].float() / 8.0)
        avg.eval()
        net = avg
        log(f"resfast: SWA over {n_avg} snapshots ({swa_steps} extra steps), BN recomputed")
    log(f"resfast {stem} v{ver}: offline {steps} steps lpol {lp:.3f} lval {lv:.3f} ({time.time() - t:.0f}s)")
    out = MODELS / f"{stem}_v{ver}.onnx"
    export(net, out)
    ok, msg = validate(out)
    s = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    xb = x[:8]
    t = time.perf_counter()
    for _ in range(200):
        s.run(None, {"observation": xb})
    ms = (time.perf_counter() - t) / 200 * 1000
    log(f"resfast {stem} v{ver}: exported {out.name} graph {graph_size(out)} {ms:.2f} ms/call b8, validator "
        f"{'ACCEPTED' if ok else 'REJECTED'}")
    if not ok:
        print(msg)
    RCKPT.mkdir(exist_ok=True)
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "best": (str(HAND), 0.5)},
               RCKPT / f"{stem}_v{ver}.pt")
