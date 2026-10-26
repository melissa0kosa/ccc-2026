"""Shared: data loading, D4 symmetry maps, PriorNet (hand_v3 FastNet + zero-init policy delta)."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
L4 = HERE.parents[1]
HAND = L4 / "agents" / "hand"
for p in (str(L4), str(HAND)):
    if p not in sys.path:
        sys.path.insert(0, p)
from fast import FastNet  # noqa: E402  (read-only import)


# ----------------------------------------------------------------------------- symmetry
def perms():
    P = np.zeros((8, 64), np.int64)
    for s in range(8):
        for sq in range(64):
            r, c = divmod(sq, 8)
            if s & 4:
                r, c = c, r
            if s & 1:
                c = 7 - c
            if s & 2:
                r = 7 - r
            P[s, sq] = r * 8 + c
    return P


PERM = perms()                                                   # sq -> PERM[s, sq]
IDXMAP = (PERM[:, :, None] * 64 + PERM[:, None, :]).reshape(8, 4096)
INV = np.argsort(PERM, axis=1)                                   # gather index for planes
INVIDX = np.argsort(IDXMAP, axis=1)                              # gather index for 4096 vectors


def tf_obs_np(obs, s):
    flat = obs.reshape(16, 64)
    return flat[:, INV[s]].reshape(16, 8, 8)


# ----------------------------------------------------------------------------- data
def load_games(files):
    games = []
    for f in files:
        with open(f, "rb") as fh:
            while True:
                try:
                    games.append(pickle.load(fh))
                except EOFError:
                    break
                except Exception:
                    break  # truncated tail of a file still being written
    return games


def games_to_arrays(games, tsharp: float = 1.0):
    obs, tgt, mask, move, best = [], [], [], [], []
    for g in games:
        for r in g["recs"]:
            idx = r["idx"].astype(np.int64)
            cnt = r["cnt"].astype(np.float64)
            if len(idx) == 0 or cnt.sum() <= 0:
                continue
            t = np.zeros(4096, np.float32)
            w = cnt ** (1.0 / tsharp)
            t[idx] = w / w.sum()
            m = np.zeros(4096, bool)
            m[idx] = True
            obs.append(r["obs"])
            tgt.append(t)
            mask.append(m)
            move.append(r["move"])
            best.append(idx[int(np.argmax(cnt))])  # lowest index on ties (idx sorted) = referee
    return (np.stack(obs).astype(np.float32), np.stack(tgt), np.stack(mask),
            np.array(move), np.array(best))


# ----------------------------------------------------------------------------- model
def load_fast(name="w_v3.json"):
    d = json.loads((HAND / "data" / name).read_text())
    return FastNet(d["weights"], d.get("bias", 0.0), d.get("pol"))


class PriorNet(nn.Module):
    """policy = base_logits * exp(s) + delta(trunk); value = FastNet value (untouched)."""

    def __init__(self, fast: FastNet, ch: int = 32, blocks: int = 3, rank: int = 8):
        super().__init__()
        self.fast = fast
        for p in self.fast.parameters():
            p.requires_grad_(False)
        cin = 16 + 11 + 28
        self.stem = nn.Conv2d(cin, ch, 3, padding=1)
        self.convs = nn.ModuleList()
        for _ in range(blocks):
            self.convs.append(nn.Conv2d(ch, ch, 3, padding=1))
            self.convs.append(nn.Conv2d(ch, ch, 3, padding=1))
        self.head = nn.Conv2d(ch, 64, 1)            # channel = from-square, spatial = to-square
        self.bl_from = nn.Conv2d(ch, rank, 1)
        self.bl_to = nn.Conv2d(ch, rank, 1)
        self.log_t = nn.Parameter(torch.zeros(1))
        self.rank = rank
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.bl_to.weight); nn.init.zeros_(self.bl_to.bias)

    def feats(self, x):
        f = self.fast
        base = F.conv2d(x, f.w_base, f.b_base)
        empty = base[:, 0:1]
        x0 = F.conv2d(x, f.w_sel)
        r1 = F.conv2d(x0, f.w_shift, padding=1, groups=80)
        r2 = F.conv2d(r1 * empty, f.w_shift, padding=1, groups=80)
        r3 = F.conv2d(r2 * empty, f.w_shift, padding=1, groups=80)
        sl = F.conv2d(torch.cat([r1, r2, r3], 1), f.w_comb)
        jk = F.conv2d(x, f.w_jk, padding=2)
        R = F.conv2d(torch.cat([sl, jk], 1), f.w_reorg)
        return torch.cat([x, base, R], 1)

    def delta(self, x):
        h = F.relu(self.stem(self.feats(x)))
        for i in range(0, len(self.convs), 2):
            y = F.relu(self.convs[i](h))
            h = F.relu(h + self.convs[i + 1](y))
        d = self.head(h).flatten(1)                                  # (B,4096) idx = from*64+to
        a = self.bl_from(h).reshape(-1, self.rank, 64)                             # (B,r,64)
        b = self.bl_to(h).reshape(-1, self.rank, 64)
        d = d + torch.matmul(a.transpose(1, 2), b).flatten(1)        # [f,t] = sum_k a_k(f) b_k(t)
        return d

    def forward(self, x):
        logits, v = self.fast(x)
        return logits * torch.exp(self.log_t) + self.delta(x), v
