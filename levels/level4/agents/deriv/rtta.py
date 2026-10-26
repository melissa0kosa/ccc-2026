"""Residual-only symmetry averaging on az_res4f_v2: base run once on x, residual trunk on [x, sym(x)].

policy = p_h(x) + mean_s map_s(dp(sym_s x)), value = tanh(v_h(x) + mean_s dv(sym_s x; S,G of x))

uv run python levels/level4/agents/deriv/rtta.py [syms e.g. 0,1] [out_name]
"""
if __name__ == "__main__":
    import sys
    from pathlib import Path

    HERE = Path(__file__).resolve().parent
    L4 = HERE.parents[1]
    sys.path.insert(0, str(L4 / "agents" / "az"))
    sys.path.insert(0, str(L4))
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import reslib
    from derive import INV, PERM, IDX, tf_obs
    from c1 import export_onnx

    syms = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0,1").split(",")]
    name = sys.argv[2] if len(sys.argv) > 2 else "rtta" + str(len(syms))
    assert syms[0] == 0
    r = reslib.ResFast(32, 2, kind="v4f")
    r.load_state_dict(torch.load(L4 / "agents" / "az" / "ckpt_res" / "az_res4f_v2.pt", map_location="cpu")["net"])
    r.eval()
    k = len(syms)

    class RTTA(nn.Module):
        def __init__(self, r):
            super().__init__()
            self.r = r
            for s in syms[1:]:
                self.register_buffer(f"inv{s}", torch.from_numpy(INV[s].copy()))
                self.register_buffer(f"perm{s}", torch.from_numpy(PERM[s].copy()))

        def forward(self, x):
            r = self.r
            ph, vh, S, G = r.base(x)
            vh = vh.reshape(-1)
            xf = x.reshape(-1, 16, 64)
            xs = [x] + [xf.index_select(2, getattr(self, f"inv{s}")).reshape(-1, 16, 8, 8) for s in syms[1:]]
            xx = torch.cat(xs, 0)
            h = r.blocks(r.stem(xx))
            f = r.pf(h).reshape(-1, r.k, 64).transpose(1, 2)
            t = r.pt(h).reshape(-1, r.k, 64)
            dp = (torch.matmul(f, t) * r.scale + r.p64(h).reshape(-1, 64, 64)).reshape(k, -1, 64, 64)
            acc = dp[0]
            for i, s in enumerate(syms[1:], 1):
                p = getattr(self, f"perm{s}")
                acc = acc + dp[i].index_select(1, p).index_select(2, p)
            SG = r.fbn(torch.cat([S, G], 1))
            z = torch.cat([r.v1(h).flatten(1), h.mean(dim=(2, 3)), torch.cat([SG] * k, 0)], 1)
            dv = r.vfc2(F.relu(r.vfc1(z))).reshape(k, -1).mean(0)
            return ph + (acc * (1.0 / k)).reshape(-1, 4096), torch.tanh(vh + dv)

    m = RTTA(r).eval()
    out = L4 / "models" / f"deriv_4f2_{name}.onnx"
    export_onnx(m, out, check=False)
    import onnx
    g = onnx.load(str(out)).graph
    print(out.name, len(g.node), len(g.initializer))

    # ---- checks
    import pickle
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    games = pickle.load(open(L4 / "agents" / "az" / "data" / "games_end_holdout1.pkl", "rb"))
    games += pickle.load(open(L4 / "agents" / "az" / "data" / "games_res_v21_66852500.pkl", "rb"))[:30]
    x = np.concatenate([gm["obs"] for gm in games]).astype(np.float32) / 8
    C = ort.InferenceSession(str(L4 / "models" / "az_res4f_v2.onnx"), so, providers=["CPUExecutionProvider"])
    D = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ph, vh, S, G = r.base(torch.from_numpy(x))
        refs = []
        for s in syms:
            xs = torch.from_numpy(tf_obs(x, s).copy())
            ph2, vh2, S2, G2 = r.base(xs)
            print(f"sym {s}: base invariance |dS| {float((S2 - S).abs().max()):.2e} |dG| {float((G2 - G).abs().max()):.2e}"
                  f" |dvh| {float((vh2.reshape(-1) - vh.reshape(-1)).abs().max()):.2e}"
                  f" |dph mapped| {float((ph2.numpy()[:, IDX[s]] - ph.numpy()).max()):.2e}")
            pp, vv = C.run(None, {"observation": tf_obs(x, s).copy()})
            refs.append((pp[:, IDX[s]] - ph2.numpy()[:, IDX[s]], np.arctanh(np.clip(vv, -1 + 1e-7, 1 - 1e-7)) - vh2.numpy().reshape(-1)))
    dp_ref = np.mean([a for a, _ in refs], 0)
    dv_ref = np.mean([b for _, b in refs], 0)
    pd, vd = D.run(None, {"observation": x})
    pc, vc = C.run(None, {"observation": x})
    print("parity vs reference: |dpol|", np.abs(pd - (ph.numpy() + dp_ref)).max(),
          "|dval|", np.abs(vd - np.tanh(vh.numpy().reshape(-1) + dv_ref)).max())
    # how often the argmax over legal moves changes vs the champion
    from c1 import encoding, CYAN, MAGENTA
    from colony_one.board import Board
    ch = 0; n = 0
    for i in range(0, len(x), 3):
        o = x[i]; team = MAGENTA if o[15, 0, 0] > 0.5 else CYAN
        msk = encoding.legal_action_mask(Board.from_observation(o, team), team)
        if msk.sum() < 2:
            continue
        n += 1
        ch += int(np.argmax(np.where(msk, pc[i], -1e9)) != np.argmax(np.where(msk, pd[i], -1e9)))
    print(f"prior argmax changed in {ch}/{n} positions; mean |dv| {np.abs(vd - vc).mean():.3f}")
