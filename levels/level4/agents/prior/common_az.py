"""AzPrior: frozen az_res_v6 (ResModel on hand_v2, ckpt agents/az/ckpt_res/az_res_v6.pt) + zero-init policy delta.

policy = (p_h + dp) * exp(log_t) + delta(cat[x, h_az]);  value = az_res_v6 value (untouched).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
AZ = HERE.parents[0] / "az"
if str(AZ) not in sys.path:
    sys.path.insert(0, str(AZ))
import reslib  # noqa: E402  (read-only import)


def load_az(ver="az_res_v6"):
    ck = torch.load(AZ / "ckpt_res" / f"{ver}.pt", weights_only=False, map_location="cpu")
    m = reslib.ResModel(32, 2)
    m.load_state_dict(ck["net"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class AzPrior(nn.Module):
    def __init__(self, az: nn.Module, ch: int = 32, blocks: int = 2, rank: int = 8):
        super().__init__()
        self.az = az
        self.rank = rank
        self.stem = nn.Conv2d(16 + 32, ch, 3, padding=1)
        self.convs = nn.ModuleList()
        for _ in range(blocks):
            self.convs.append(nn.Conv2d(ch, ch, 3, padding=1))
            self.convs.append(nn.Conv2d(ch, ch, 3, padding=1))
        self.head = nn.Conv2d(ch, 64, 1)
        self.bl_from = nn.Conv2d(ch, rank, 1)
        self.bl_to = nn.Conv2d(ch, rank, 1)
        self.log_t = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.bl_to.weight); nn.init.zeros_(self.bl_to.bias)

    def train(self, mode=True):
        super().train(mode)
        self.az.eval()  # frozen BN statistics
        return self

    def forward(self, x):
        a = self.az
        feats, m = a.base.features(x)
        vh = a.base.lin(feats).reshape(-1)
        ph = a.base.policy(m)
        h = a.blocks(a.stem(x))
        f = a.pf(h).reshape(-1, a.k, 64).transpose(1, 2)
        t = a.pt(h).reshape(-1, a.k, 64)
        dp = (torch.matmul(f, t) * a.scale + a.p64(h).reshape(-1, 64, 64)).reshape(-1, 4096)
        z = torch.cat([a.v1(h).flatten(1), h.mean(dim=(2, 3)), feats], 1)
        dv = a.vfc2(F.relu(a.vfc1(z))).reshape(-1)
        v = torch.tanh(vh + dv)
        g = F.relu(self.stem(torch.cat([x, h], 1)))
        for i in range(0, len(self.convs), 2):
            y = F.relu(self.convs[i](g))
            g = F.relu(g + self.convs[i + 1](y))
        d = self.head(g).flatten(1)
        fa = self.bl_from(g).reshape(-1, self.rank, 64)
        tb = self.bl_to(g).reshape(-1, self.rank, 64)
        d = d + torch.matmul(fa.transpose(1, 2), tb).reshape(-1, 4096)
        return (ph + dp) * torch.exp(self.log_t) + d, v
