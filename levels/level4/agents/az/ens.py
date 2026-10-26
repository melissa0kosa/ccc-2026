"""Ensemble of residual corrections on one shared FastNet base: policy = p_h + mean(dp_i), value = tanh(v_h + mean(dv_i)).

uv run python levels/level4/agents/az/ens.py OUT_NAME ckpt1.pt ckpt2.pt [...]
"""
if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np
    import onnxruntime as ort
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import reslib
    from reslib import MODELS, RCKPT, export, graph_size, log

    class Ens(nn.Module):
        def __init__(self, members):
            super().__init__()
            self.base = members[0].base
            self.m = nn.ModuleList(members)

        def forward(self, x):
            ph, vh, S, G = self.base(x)
            vh = vh.reshape(-1)
            SG = torch.cat([S, G], 1)
            dps, dvs = [], []
            for r in self.m:
                h = r.blocks(r.stem(x))
                f = r.pf(h).reshape(-1, r.k, 64).transpose(1, 2)
                t = r.pt(h).reshape(-1, r.k, 64)
                dps.append(torch.matmul(f, t) * r.scale + r.p64(h).reshape(-1, 64, 64))
                z = torch.cat([r.v1(h).flatten(1), h.mean(dim=(2, 3)), r.fbn(SG)], 1)
                dvs.append(r.vfc2(F.relu(r.vfc1(z))))
            n = float(len(self.m))
            dp = sum(dps) * (1.0 / n)
            dv = sum(dvs) * (1.0 / n)
            return ph + dp.reshape(-1, 4096), torch.tanh(vh + dv.reshape(-1))

    out = sys.argv[1]
    members = []
    for c in sys.argv[2:]:
        r = reslib.ResFast(32, 2)
        r.load_state_dict(torch.load(c, map_location="cpu")["net"])
        members.append(r.eval())
    ens = Ens(members).eval()
    p = MODELS / f"{out}.onnx"
    export(ens, p)
    # parity: ensemble of one == member; check vs members' own exports on random real-ish positions
    x = (np.random.rand(16, 16, 8, 8) > 0.9).astype(np.float32)
    with torch.no_grad():
        a = ens(torch.from_numpy(x))
        mp = [m(torch.from_numpy(x)) for m in members]
    exp_p = sum(q[0] for q in mp) / len(mp)
    print("torch check max|dpolicy| vs manual mean", float((a[0] - exp_p).abs().max()))
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
    o = s.run(None, {"observation": x})
    print("onnx vs torch", float(np.abs(o[0] - a[0].numpy()).max()), float(np.abs(o[1] - a[1].numpy()).max()))
    t = time.perf_counter()
    for _ in range(200):
        s.run(None, {"observation": x[:8]})
    log(f"ens {out}: members {sys.argv[2:]} graph {graph_size(p)} {(time.perf_counter() - t) / 200 * 1000:.2f} ms/call b8")
