"""Convert teacher-labelled endgame npz files to the sp_v4f.pkl record format:
(planes 0-12 uint8 (13,8,8), planes 13-15 float32 (3,), z, indices int16, counts int16 (policy*200), root Q)."""
import glob
import pickle
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
T = 0.4
recs = []
src = {}
for pat in sys.argv[1:] or ["eg1_*.npz", "eg2_*.npz", "egval_*.npz"]:
    for f in sorted(glob.glob(str(HERE / "data" / pat))):
        d = np.load(f)
        obs, li, sc, best = d["obs"], d["li"], d["sc"], d["best"]
        for i in range(len(obs)):
            m = li[i] >= 0
            idx = li[i][m].astype(np.int16)
            s = sc[i][m].astype(np.float64)
            p = np.exp((np.clip(s, -30, 30) - np.clip(s, -30, 30).max()) / T)
            p /= p.sum()
            cnt = np.round(p * 200).astype(np.int16)
            cnt[np.argmax(p)] = max(cnt[np.argmax(p)], 1)
            keep = cnt > 0
            q = float(np.tanh(best[i] / 3.0))
            z = float(np.sign(best[i])) if abs(best[i]) >= 50 else q
            recs.append((obs[i][:13].astype(np.uint8), obs[i][13:16, 0, 0].astype(np.float32), z,
                         idx[keep], cnt[keep], q))
        src[Path(f).name] = len(obs)
out = HERE / "data" / "endgame_labels.pkl"
pickle.dump(recs, open(out, "wb"))
qs = np.array([r[5] for r in recs])
mag = np.array([r[1][2] > 0.5 for r in recs])
print(f"saved {len(recs)} records to {out}; sources {src}")
print(f"magenta-to-move {mag.mean():.2f}; |Q|>0.95 (decided) {np.mean(np.abs(qs) > 0.95):.2f}; mean Q {qs.mean():+.3f}")
