"""Value calibration on endgame positions: model value vs deep-teacher tanh(best/3).
uv run python levels/level4/agents/distill/calib.py "eg1_*.npz" model1.onnx model2.onnx ..."""
import glob
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
HERE = Path(__file__).resolve().parent
files = sorted(glob.glob(str(HERE / "data" / sys.argv[1])))
d = [np.load(f) for f in files]
obs = np.concatenate([x["obs"] for x in d]).astype(np.float32)
best = np.concatenate([x["best"] for x in d])
t = np.tanh(best / 3.0)
mag = obs[:, 15, 0, 0] > 0.5
nontriv = np.abs(best) < 50
print(f"{len(obs)} positions ({nontriv.sum()} non-trivial), magenta-to-move {mag.mean():.2f}")
for m in sys.argv[2:]:
    s = ort.InferenceSession(m, providers=["CPUExecutionProvider"])
    v = np.concatenate([s.run(None, {"observation": obs[i:i + 512]})[1] for i in range(0, len(obs), 512)])
    def stats(mask):
        e = np.abs(v[mask] - t[mask]).mean()
        sg = (np.sign(v[mask]) == np.sign(t[mask]))[np.abs(t[mask]) > 0.3].mean()
        return f"MAE {e:.3f} sign-agree {sg:.2f}"
    print(f"{Path(m).name:24s} all: {stats(np.ones(len(v), bool))} | cyan: {stats(~mag)} | magenta: {stats(mag)} | "
          f"mean v-t mag {np.mean(v[mag]-t[mag]):+.2f} cyan {np.mean(v[~mag]-t[~mag]):+.2f}")
