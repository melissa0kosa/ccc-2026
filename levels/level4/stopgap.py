"""Stopgap Level 4 model: material + kernel firewall danger + guards + reach threats, capture prior.

All features are fixed-weight convolutions (validator-clean). Run with PyCharm ▶ or:
    uv run python levels/level4/stopgap.py [games]
Writes levels/level4/models/stopgap_v1.onnx and arenas it vs the current seat.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1 import MODELS, arena, export_onnx  # noqa: E402

# planes: 0-5 own Ping, Daemon, Exploit, Trace, Ice, Kernel; 6-11 enemy; 12 locked; 13 actions; 14 urgency
VALUES = np.array([15, 45, 25, 30, 50, 100], np.float32)
R = 3  # reach kernels are 7x7, centre (3,3)


def reach(piece: str) -> np.ndarray:
    k = np.zeros((7, 7), np.float32)
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            d = max(abs(dr), abs(dc))
            diag = abs(dr) == abs(dc) and d > 0
            orth = (dr == 0) != (dc == 0)
            if piece == "P" and diag and d <= 2:
                k[dr + R, dc + R] = 1
            if piece == "D" and (d == 1 or (orth and d <= 3)):
                k[dr + R, dc + R] = 1
            if piece == "E" and sorted((abs(dr), abs(dc))) == [1, 2]:
                k[dr + R, dc + R] = 1
            if piece == "T" and diag and 2 <= d <= 3:
                k[dr + R, dc + R] = 1
            if piece == "I" and (diag or orth) and d <= 2:
                k[dr + R, dc + R] = 1
            if piece == "K" and d == 1:
                k[dr + R, dc + R] = 1
    return k


class Stopgap(nn.Module):
    def __init__(self, mat=1 / 40, w_danger=1.5, w_center=0.25, w_guard=0.35, w_thr=0.25,
                 w_hang=0.3, prior_cap=1 / 20, prior_escape=1.5):
        super().__init__()
        self.c = dict(mat=mat, w_danger=w_danger, w_center=w_center, w_guard=w_guard,
                      w_thr=w_thr, w_hang=w_hang, prior_cap=prior_cap, prior_escape=prior_escape)
        # 1x1 picks: own kernel, enemy kernel, locked, urgency, own value map, enemy value map,
        # own non-kernel, enemy non-kernel
        pick = np.zeros((8, 16, 1, 1), np.float32)
        pick[0, 5] = pick[1, 11] = pick[2, 12] = pick[3, 14] = 1
        pick[4, :6, 0, 0] = VALUES
        pick[5, 6:12, 0, 0] = VALUES
        pick[6, :5] = pick[7, 6:11] = 1
        self.pick = nn.Conv2d(16, 8, 1, bias=False)
        self.pick.weight.data = torch.tensor(pick)
        # reach maps: channel 0 = squares attacked by own pieces, 1 = by enemy pieces
        rk = np.zeros((2, 16, 7, 7), np.float32)
        for i, p in enumerate("PDETIK"):
            rk[0, i] = rk[1, 6 + i] = reach(p)
        self.reach = nn.Conv2d(16, 2, 7, padding=3, bias=False)
        self.reach.weight.data = torch.tensor(rk)
        # 3x3 ones on [locked, own non-kernel, enemy non-kernel]
        self.adj = nn.Conv2d(3, 3, 3, padding=1, bias=False, groups=3)
        self.adj.weight.data = torch.ones(3, 1, 3, 3)
        ring = np.array([[min(r, c, 7 - r, 7 - c) for c in range(8)] for r in range(8)], np.float32)
        self.register_buffer("ring0", torch.tensor(ring == 0, dtype=torch.float32).reshape(1, 1, 8, 8))
        self.register_buffer("center", torch.tensor(ring / 3).reshape(1, 1, 8, 8))
        self.register_buffer("zeros64", torch.zeros(64, 1))

    def forward(self, x):
        c = self.c
        p = self.pick(x)
        own_k, en_k, locked, urg = p[:, 0:1], p[:, 1:2], p[:, 2:3], p[:, 3:4]
        own_val, en_val, own_nk, en_nk = p[:, 4:5], p[:, 5:6], p[:, 6:7], p[:, 7:8]
        a = self.adj(torch.cat([locked, own_nk, en_nk], 1))
        danger = torch.clamp(a[:, 0:1] + self.ring0, 0, 1) * (1 - locked)
        own_guard = torch.clamp(a[:, 1:2], 0, 1)
        en_guard = torch.clamp(a[:, 2:3], 0, 1)
        r = self.reach(x)
        own_att, en_att = torch.clamp(r[:, 0:1], 0, 1), torch.clamp(r[:, 1:2], 0, 1)

        def s(t):
            return t.flatten(1).sum(1)

        v = c["mat"] * s(own_val - en_val)
        v = v - c["w_danger"] * s(urg * danger * (own_k - en_k))
        v = v + c["w_center"] * s(self.center * (own_k - en_k))
        v = v + c["w_guard"] * s(own_k * own_guard - en_k * en_guard)
        v = v - c["w_thr"] * s(own_k * en_att * (1 - own_guard)) + c["w_thr"] * s(en_k * own_att * (1 - en_guard))
        v = v + c["w_hang"] * c["mat"] * s(en_val * own_att - 0.5 * own_val * en_att)
        value = torch.tanh(v)

        to_term = c["prior_cap"] * en_val - 0.5 * danger * urg          # capture big pieces, avoid fire
        from_term = c["prior_escape"] * own_k * danger * urg            # move a kernel standing in fire
        policy = from_term.reshape(-1, 64, 1) + to_term.reshape(-1, 1, 64)
        return policy.reshape(-1, 4096), value


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    path = export_onnx(Stopgap(), MODELS / "stopgap_v1.onnx")
    print("exported + validated", path)
    arena(path, MODELS / "seat_l3_material.onnx", games=games, workers=4)
    arena(path, "greedy", games=games, workers=4)
