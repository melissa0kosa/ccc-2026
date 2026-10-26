"""Offline experiment: train a net on saved game data, arena vs stopgap with 1 worker.

uv run python levels/level4/agents/az/exp.py NAME STEPS QMIX PATTERN [init_ckpt] [games] [channels] [blocks]
"""
if __name__ == "__main__":
    import pickle
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch

    import azlib
    from azlib import DATA, HERE, STOPGAP, Buffer, Net, arena, export, train

    name, steps, qmix, pattern = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    init = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else ""
    games = int(sys.argv[6]) if len(sys.argv) > 6 else 60
    ch = int(sys.argv[7]) if len(sys.argv) > 7 else 32
    bl = int(sys.argv[8]) if len(sys.argv) > 8 else 4
    torch.set_num_threads(2)
    buf = Buffer(10**9)
    for f in sorted(DATA.glob(pattern), key=lambda q: q.stat().st_mtime):
        with open(f, "rb") as fh:
            for g in pickle.load(fh):
                buf.add(g)
    print("samples", buf.n(), flush=True)
    net = Net(ch, bl).cuda()
    if init:
        net.load_state_dict(torch.load(init, map_location="cuda")["net"])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    t = time.time()
    lp, lv, n = train(net, opt, buf, steps, qmix=qmix)
    print(f"train {steps} steps lpol {lp:.3f} lval {lv:.3f} {time.time() - t:.0f}s", flush=True)
    out = HERE / "exp" / f"{name}.onnx"
    out.parent.mkdir(exist_ok=True)
    export(net, out)
    torch.save({"net": net.state_dict()}, HERE / "exp" / f"{name}.pt")
    arena(out, STOPGAP, games, workers=1)
