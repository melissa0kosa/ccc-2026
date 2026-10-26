import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, load_games, games_to_arrays  # noqa
from common_az import load_az, AzPrior  # noqa
from c1 import evaluator, MODELS  # noqa
games = load_games(sorted((HERE / "data").glob("gen_r1_11.pkl")))[:5]
X = games_to_arrays(games)[0][:64]
M = games_to_arrays(games)[2][:64]
m = AzPrior(load_az()).eval()
with torch.no_grad():
    p, v = m(torch.from_numpy(X))
po, vo = evaluator(str(MODELS / "az_res_v6.onnx"))(X)
p = p.numpy(); po = np.asarray(po)
print("dv", np.abs(v.numpy() - np.asarray(vo)).max(), "dp legal", max(np.abs((p[i][M[i]] - p[i][M[i]].mean()) - (po[i][M[i]] - po[i][M[i]].mean())).max() for i in range(len(X))))
