"""Scratch: hand_v2 features/value at the child positions of key live decisions (from the child's side to move)."""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "reviews")); sys.path.insert(0, str(HERE / "agents/hand"))
import hand_v2_src as H
from c1 import evaluator, MODELS
from analyze_matches import parse
from sg_pos import env_at
d = json.loads((HERE / "agents/hand/data/w_v2.json").read_text())
net = H.build(d["weights"], d["bias"], d["pol"]); HV = evaluator(str(MODELS / "hand_v2.onnx"))
ms = {m["match_id"]: m for m in json.loads((HERE / "data/matches/matches_hand_v2_1.json").read_text())}
W = np.array([d["weights"].get(n, 0.0) for n in H.FEATURES])
def child(mid, k, mv):
    moves, _ = parse(ms[mid]["protocol"]); env = env_at(moves, k)
    a = [x for x in env.legal_actions() if env.format_action(x).replace(" ", "") == mv][0]; env.step(a); return env
for mid, k, mv, keys in [(6032, 23, "E5->C3", ["en_att_I", "en_hang_I", "en_att2_I", "en_I", "own_X", "win2"]),(6032, 22, "G7->E5", ["en_att_I", "en_hang_I", "en_att2_I", "own_I"]),
                         (6032, 22, "H6->G6", ["en_att_I", "en_hang_I"]),
                         (6032, 35, "G7->G6", ["t2_en", "enK_thr", "g_en", "enK_att", "en_att_T", "en_hang_T", "en_att2_T"]),
                         (6032, 35, "F7->E7", ["t2_en", "enK_thr", "g_en", "enK_att", "en_att_T", "en_hang_T"])]:
    env = child(mid, k, mv); x = torch.tensor(env.observation()[None], dtype=torch.float32)
    with torch.no_grad():
        f, _ = net.features(x); _, v = net(x)
    f = f[0].numpy(); vo = HV(env.observation()[None])[1][0]
    print(f"{mid} after a{k+1} {mv}: {env.current_team} to move al={env.actions_left}  v(torch)={v.item():+.3f} v(onnx)={vo:+.3f} | " +
          ", ".join(f"{n}={f[H.FEATURES.index(n)]:.0f}(w{d['weights'].get(n,0):+.2f})" for n in keys))
