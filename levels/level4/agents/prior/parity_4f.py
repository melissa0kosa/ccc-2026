import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, load_games, games_to_arrays  # noqa
from common_4f import load_4f, Prior4f  # noqa
from c1 import evaluator, MODELS  # noqa
games = load_games(sorted((HERE / "data").glob("gen_r1_11.pkl")))[:5]
X, _, M, _, _ = games_to_arrays(games)
X = X[:64]; M = M[:64]
m = Prior4f(load_4f()).eval()
with torch.no_grad():
    p, v = m(torch.from_numpy(X))
po, vo = evaluator(str(MODELS / "az_res4f_v2.onnx"))(X)
p = p.numpy(); po = np.asarray(po)
print("dv", np.abs(v.numpy() - np.asarray(vo)).max(), "dp legal", max(np.abs((p[i][M[i]] - p[i][M[i]].mean()) - (po[i][M[i]] - po[i][M[i]].mean())).max() for i in range(len(X))))
