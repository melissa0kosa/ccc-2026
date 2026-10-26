"""Value calibration on held-out endgame-start positions (Kernel + <=1 piece per side).
uv run python levels/level4/agents/az/calib.py model1.onnx [model2.onnx ...]"""
if __name__ == "__main__":
    import pickle
    import sys
    from pathlib import Path

    import numpy as np
    import onnxruntime as ort

    HERE = Path(__file__).resolve().parent
    games = []
    for f in sorted((HERE / "data").glob("games_end_holdout*.pkl")):
        games += pickle.load(open(f, "rb"))
    x = np.concatenate([g["obs"] for g in games]).astype(np.float32) / 8.0
    z = np.concatenate([g["z"] for g in games])
    mag = x[:, 15, 0, 0] > 0.5
    print(f"{len(games)} games, {len(x)} positions (magenta-to-move {mag.mean():.0%})")
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    for m in sys.argv[1:]:
        s = ort.InferenceSession(m, so, providers=["CPUExecutionProvider"])
        v = np.concatenate([s.run(None, {"observation": x[i:i + 256]})[1] for i in range(0, len(x), 256)])
        out = [Path(m).stem]
        for name, sel in (("MAG", mag), ("CYAN", ~mag)):
            lost = sel & (z < 0)
            won = sel & (z > 0)
            out.append(f"{name}: lost mean v {v[lost].mean():+.2f} (n={lost.sum()}), won {v[won].mean():+.2f} (n={won.sum()}), "
                       f"mse {((v[sel] - z[sel]) ** 2).mean():.3f}, sign-acc {(np.sign(v[sel]) == z[sel]).mean():.2f}")
        print(" | ".join(out))
