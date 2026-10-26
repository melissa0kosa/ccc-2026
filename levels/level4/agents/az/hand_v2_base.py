"""Hand-designed feature evaluator for COLONY: ONE (Level 4), exportable to ONNX.

Planes (canonical, side to move = "own"):
 0-5 own Ping, Daemon, Exploit, Trace, Ice, Kernel; 6-11 enemy; 12 locked;
 13 actions_left/2; 14 firewall urgency; 15 Magenta-to-move.

HandNet(params) -> (policy logits (B,4096), value (B,)).
Features are fixed (exact capture-threat maps incl. blocking via ray propagation);
value = tanh(w . features + b); policy = additive/outer-product hand prior.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
L4 = HERE.parents[1]
if str(L4) not in sys.path:
    sys.path.insert(0, str(L4))

PING, DAEMON, EXPLOIT, TRACE, ICE, KERNEL = range(6)
ORTH = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL8 = ORTH + DIAG
KNIGHT = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
SLIDERS = [PING, DAEMON, TRACE, ICE]  # ray pieces
# capture distances per (slider, direction-set)
CAP = {
    PING: (DIAG, (1, 2)),
    DAEMON: (ORTH, (2, 3)),   # + adjacent shot in all 8 handled separately (k=1 all8)
    TRACE: (DIAG, (2, 3)),
    ICE: (ALL8, (1, 2)),
}

FEATURES: list[str] = []


def _ring_masks():
    m = np.zeros((4, 8, 8), np.float32)
    for r in range(8):
        for c in range(8):
            m[min(r, c, 7 - r, 7 - c), r, c] = 1
    return m


class HandNet(nn.Module):
    def __init__(self, n_feat: int | None = None):
        super().__init__()
        # --- ray shift conv: channels = 2 sides x 4 sliders x 8 dirs = 64, grouped
        # ray channels: 0..63 = 2 sides x 4 sliders x 8 dirs; 64..79 = 2 sides' kernel x 8 dirs
        w = np.zeros((80, 1, 3, 3), np.float32)
        sel = np.zeros((80, 16, 1, 1), np.float32)
        for s in range(2):
            for i, p in enumerate(SLIDERS):
                for d, (dr, dc) in enumerate(ALL8):
                    ch = (s * 4 + i) * 8 + d
                    w[ch, 0, 1 - dr, 1 - dc] = 1  # out[t] = x[t - d]
                    sel[ch, s * 6 + p, 0, 0] = 1
            for d, (dr, dc) in enumerate(ALL8):
                ch = 64 + s * 8 + d
                w[ch, 0, 1 - dr, 1 - dc] = 1
                sel[ch, s * 6 + KERNEL, 0, 0] = 1
        self.register_buffer("w_shift", torch.tensor(w))
        self.register_buffer("w_sel", torch.tensor(sel))
        MOVE = {PING: (ORTH, (1, 2)), DAEMON: (ORTH, (1, 2, 3)), TRACE: (ORTH, (1,)), ICE: (ALL8, (1, 2))}
        # combine R1,R2,R3 (240 ch): 0..7 attack (side,slider), 8..15 move rays (side,slider),
        # 16..23 reverse attack from kernel of side s by attacker slider type i
        comb = np.zeros((24, 240, 1, 1), np.float32)
        for k in (1, 2, 3):
            for s in range(2):
                for i, p in enumerate(SLIDERS):
                    dirs, ks = CAP[p]
                    mdirs, mks = MOVE[p]
                    for d, dd in enumerate(ALL8):
                        ch = (k - 1) * 80 + (s * 4 + i) * 8 + d
                        kch = (k - 1) * 80 + 64 + s * 8 + d
                        use = (dd in dirs and k in ks) or (p == DAEMON and k == 1)
                        if use:
                            comb[s * 4 + i, ch, 0, 0] = 1
                            comb[16 + s * 4 + i, kch, 0, 0] = 1
                        if dd in mdirs and k in mks:
                            comb[8 + s * 4 + i, ch, 0, 0] = 1
        self.register_buffer("w_comb", torch.tensor(comb))
        # 5x5 conv: own X atk, own K atk, en X atk, en K atk, knight-from-own-K, knight-from-en-K
        jk = np.zeros((6, 16, 5, 5), np.float32)
        for s in range(2):
            for dr, dc in KNIGHT:
                jk[s * 2 + 0, s * 6 + EXPLOIT, 2 - dr, 2 - dc] = 1
                jk[4 + s, s * 6 + KERNEL, 2 - dr, 2 - dc] = 1
            for dr, dc in ALL8:
                jk[s * 2 + 1, s * 6 + KERNEL, 2 - dr, 2 - dc] = 1
        self.register_buffer("w_jk", torch.tensor(jk))
        ring = np.zeros((1, 1, 3, 3), np.float32)
        ring[0, 0] = 1
        ring[0, 0, 1, 1] = 0
        self.register_buffer("w_adj", torch.tensor(ring))
        rm = _ring_masks()
        self.register_buffer("ringm", torch.tensor(rm).reshape(1, 4, 8, 8))
        self.register_buffer("ringm_flat", torch.tensor(rm.reshape(4, 64)))
        self.register_buffer("ring_idx", torch.tensor((rm * np.arange(4)[:, None, None]).sum(0).astype(np.float32)).reshape(1, 1, 8, 8))
        self.register_buffer("center", torch.tensor(
            (rm * np.arange(4)[:, None, None]).sum(0).astype(np.float32)).reshape(1, 1, 8, 8))
        self.n_feat = n_feat or 0
        self.lin = nn.Linear(max(1, self.n_feat), 1)
        # policy params (scalars)
        self.pp = dict()

    # ------------------------------------------------------------------ maps
    def maps(self, x):
        own = x[:, 0:6]
        en = x[:, 6:12]
        locked = x[:, 12:13]
        occ = own.sum(1, keepdim=True) + en.sum(1, keepdim=True) + locked
        empty = 1.0 - torch.clamp(occ, 0.0, 1.0)  # (B,1,8,8)
        x0 = F.conv2d(x, self.w_sel)  # (B,80,8,8)
        r1 = F.conv2d(x0, self.w_shift, padding=1, groups=80)
        r2 = F.conv2d(r1 * empty, self.w_shift, padding=1, groups=80)
        r3 = F.conv2d(r2 * empty, self.w_shift, padding=1, groups=80)
        sl = F.conv2d(torch.cat([r1, r2, r3], 1), self.w_comb)  # (B,24,8,8)
        jk = F.conv2d(x, self.w_jk, padding=2)  # own X, own K, en X, en K, knight(ownK), knight(enK)
        # per-type attack maps in PieceType order
        a_own = torch.cat([sl[:, 0:1], sl[:, 1:2], jk[:, 0:1], sl[:, 2:3], sl[:, 3:4], jk[:, 1:2]], 1)
        a_en = torch.cat([sl[:, 4:5], sl[:, 5:6], jk[:, 2:3], sl[:, 6:7], sl[:, 7:8], jk[:, 3:4]], 1)
        # quiet-move reach per type (to empty squares)
        self._mv_own = torch.cat([sl[:, 8:9], sl[:, 9:10], jk[:, 0:1], sl[:, 10:11], sl[:, 11:12], jk[:, 1:2]], 1) * empty
        self._mv_en = torch.cat([sl[:, 12:13], sl[:, 13:14], jk[:, 2:3], sl[:, 14:15], sl[:, 15:16], jk[:, 3:4]], 1) * empty
        # squares from which a piece of type t attacks the own / enemy kernel
        adjo = F.conv2d(x[:, 5:6], self.w_adj, padding=1)
        adje = F.conv2d(x[:, 11:12], self.w_adj, padding=1)
        self._rev_ownK = torch.cat([sl[:, 16:17], sl[:, 17:18], jk[:, 4:5], sl[:, 18:19], sl[:, 19:20], adjo], 1)
        self._rev_enK = torch.cat([sl[:, 20:21], sl[:, 21:22], jk[:, 5:6], sl[:, 22:23], sl[:, 23:24], adje], 1)
        return own, en, locked, empty, a_own, a_en

    def features(self, x):
        own, en, locked, empty, a_own, a_en = self.maps(x)
        al = x[:, 13, 0, 0] * 2.0          # actions left 1 or 2
        a2 = al - 1.0                       # 1 if two actions left
        u = x[:, 14, 0, 0]
        u1 = torch.relu(u - 0.875) * 8.0   # 1 iff shrink at end of this round
        mag = x[:, 15, 0, 0]
        A_own = a_own.sum(1, keepdim=True)
        A_en = a_en.sum(1, keepdim=True)
        hit_own = torch.clamp(A_own, 0, 1)  # squares I attack
        hit_en = torch.clamp(A_en, 0, 1)

        def S(t):
            return t.sum((1, 2, 3)) if t.dim() == 4 else t.sum((1, 2))

        f = []
        names = []

        def add(name, v):
            names.append(name)
            f.append(v)

        ptn = ["P", "D", "X", "T", "I"]
        for t in range(5):
            add(f"own_{ptn[t]}", S(own[:, t]))
        for t in range(5):
            add(f"en_{ptn[t]}", S(en[:, t]))
        # attacked / hanging (side to move = own)
        for t in range(5):
            o = own[:, t:t + 1]
            add(f"own_att_{ptn[t]}", S(o * hit_en))
            add(f"own_hang_{ptn[t]}", S(o * hit_en * (1 - hit_own)))
        for t in range(5):
            e = en[:, t:t + 1]
            add(f"en_att_{ptn[t]}", S(e * hit_own))
            add(f"en_hang_{ptn[t]}", S(e * hit_own * (1 - hit_en)))
            add(f"en_att2_{ptn[t]}", S(e * hit_own) * a2)
        # kernels
        ownK = own[:, 5:6]
        enK = en[:, 5:6]
        ownNK = own[:, 0:5].sum(1, keepdim=True)
        enNK = en[:, 0:5].sum(1, keepdim=True)
        adj_ownK = F.conv2d(ownK, self.w_adj, padding=1)
        adj_enK = F.conv2d(enK, self.w_adj, padding=1)
        g_own = S(adj_ownK * ownNK)
        g_en = S(adj_enK * enNK)
        ka_own = torch.clamp(S(ownK * A_en), 0, 1)   # my kernel attacked
        ka_en = torch.clamp(S(enK * A_own), 0, 1)    # enemy kernel attacked by me
        g0_en = torch.relu(1 - g_en)
        g1_en = torch.relu(1 - torch.abs(g_en - 1))
        g0_own = torch.relu(1 - g_own)
        g1_own = torch.relu(1 - torch.abs(g_own - 1))
        win_now = torch.clamp(ka_en * (g0_en + g1_en * a2), 0, 1)
        add("win_now", win_now)
        add("enK_att", ka_en)
        add("enK_att_g0", ka_en * g0_en)
        add("enK_att_g1", ka_en * g1_en)
        add("ownK_att", ka_own)
        add("ownK_att_g0", ka_own * g0_own)
        add("ownK_att_g1", ka_own * g1_own)
        add("ownK_att_n", S(ownK * A_en))
        add("enK_att_n", S(enK * A_own))
        add("g_own", torch.clamp(g_own, 0, 3))
        add("g_en", torch.clamp(g_en, 0, 3))
        add("ownK_zone_hit", S(adj_ownK * hit_en))
        add("enK_zone_hit", S(adj_enK * hit_own))
        add("mob_own", S(hit_own * (1 - locked)))
        add("mob_en", S(hit_en * (1 - locked)))
        # two-action kernel strikes: quiet move to a square that attacks the kernel
        t2_en = S(self._mv_own * self._rev_enK)     # my options vs enemy kernel
        t2_own = S(self._mv_en * self._rev_ownK)    # enemy options vs my kernel
        t2e = torch.clamp(t2_en, 0, 1)
        t2o = torch.clamp(t2_own, 0, 1)
        win2 = torch.clamp(win_now + a2 * g0_en * t2e, 0, 1)
        add("win2", win2)
        add("t2_en", torch.clamp(t2_en, 0, 3))
        add("t2_own", torch.clamp(t2_own, 0, 3))
        thr_own = torch.clamp(g0_own * torch.clamp(ka_own + t2o, 0, 1) + g1_own * ka_own, 0, 1)
        thr_en = torch.clamp(g0_en * torch.clamp(ka_en + t2e, 0, 1) + g1_en * ka_en, 0, 1)
        add("ownK_thr", thr_own)
        add("ownK_thr_a1", thr_own * (1 - a2))
        add("enK_thr", thr_en)
        add("enK_thr_a1", thr_en * (1 - a2))
        add("ownK_g0", g0_own)
        add("enK_g0", g0_en)
        # firewall: next ring to lock
        lr = (locked * self.ringm).amax((2, 3))  # (B,4) ring locked?
        nxt = torch.cat([1 - lr[:, 0:1], lr[:, 0:3] * (1 - lr[:, 1:4])], 1)  # (B,4)
        nxt2 = torch.cat([lr[:, 0:1] * 0.0, nxt[:, 0:3]], 1)
        N1 = (nxt @ self.ringm_flat).reshape(-1, 1, 8, 8)
        N2 = (nxt2 @ self.ringm_flat).reshape(-1, 1, 8, 8)
        nlock = lr.sum(1)
        okN1 = S(ownK * N1)
        ekN1 = S(enK * N1)
        add("ownK_N1_u", okN1 * u)
        add("ownK_N1_u1", okN1 * u1)
        add("ownK_N1_u1_a1", okN1 * u1 * (1 - a2))
        add("ownK_N2_u1", S(ownK * N2) * u1)
        add("enK_N1_u", ekN1 * u)
        add("enK_N1_u1_mag", ekN1 * u1 * mag)        # enemy already moved -> burns
        add("enK_N1_u1_cyan", ekN1 * u1 * (1 - mag))
        add("ownK_N1_u1_mag", okN1 * u1 * mag)
        vals = [1.0, 3.0, 1.7, 2.0, 3.0]
        ownV = sum(own[:, t:t + 1] * vals[t] for t in range(5))
        enV = sum(en[:, t:t + 1] * vals[t] for t in range(5))
        u2 = torch.clamp((u - 0.625) * 8.0, 0, 1)   # shrink within 2 rounds
        esc_own = S(adj_ownK * empty * (1 - N1))
        esc_en = S(adj_enK * empty * (1 - N1))
        add("ownK_N1_u2", okN1 * u2)
        add("ownK_N1_u1_noesc", okN1 * u1 * torch.relu(1 - esc_own))
        add("enK_N1_u2", ekN1 * u2)
        add("enK_N1_u1_noesc_cyan", ekN1 * u1 * torch.relu(1 - esc_en) * (1 - mag))
        add("ownV_N1_u1", S(ownV * N1) * u1)
        add("enV_N1_u1_mag", S(enV * N1) * u1 * mag)
        add("enV_N1_u1_cyan", S(enV * N1) * u1 * (1 - mag))
        add("ownV_N1_u", S(ownV * N1) * u)
        add("enV_N1_u", S(enV * N1) * u)
        # kernel depth relative to the open board (0 = on the next ring)
        dk_own = S(ownK * self.ring_idx) - nlock
        dk_en = S(enK * self.ring_idx) - nlock
        add("ownK_depth", dk_own)
        add("enK_depth", dk_en)
        add("Kdepth_diff_bare", (dk_own - dk_en) * torch.relu(1 - S(ownNK) - S(enNK)))
        add("mag", mag)
        add("a2", a2)
        add("mag_late", mag * u1)
        add("nlock", nlock)
        add("nlock_mag", nlock * mag)
        add("mat_x_nlock", (S(ownV) - S(enV)) * nlock)
        F_ = torch.stack(f, 1)
        if not FEATURES:
            FEATURES.extend(names)
        return F_, dict(own=own, en=en, ownV=ownV, enV=enV, A_own=A_own, A_en=A_en,
                        hit_own=hit_own, hit_en=hit_en, N1=N1, u=u, u1=u1, ownK=ownK, enK=enK,
                        adj_enK=adj_enK, adj_ownK=adj_ownK, g_en=g_en, a2=a2, empty=empty)

    def forward(self, x):
        feats, m = self.features(x)
        v = torch.tanh(self.lin(feats).reshape(-1))
        return self.policy(m), v

    # ------------------------------------------------------------------ policy
    def policy(self, m):
        P = self.pp
        B = m["own"].shape[0]
        ownV = m["ownV"] + m["ownK"] * P["k_val"]
        enV = m["enV"] + m["enK"] * P["ek_val"]
        hit_en = m["hit_en"]
        hit_own = m["hit_own"]
        u1 = m["u1"].reshape(-1, 1, 1, 1)
        N1 = m["N1"]
        # from-square terms (B,64)
        fr = (P["esc"] * ownV * hit_en * (1 - hit_own) + P["esc_def"] * ownV * hit_en * hit_own
              + P["fw_from"] * ownV * N1 * u1 + P["k_from"] * m["ownK"])
        # to-square terms (B,64)
        to = (P["cap"] * enV + P["cap_hang"] * enV * (1 - hit_en)
              + P["near_enK"] * m["adj_enK"] * m["empty"] + P["fw_to"] * N1 * u1 * (-1.0)
              + P["center"] * self.center)
        fr = fr.reshape(-1, 64, 1)
        to = to.reshape(-1, 1, 64)
        # interaction: moving piece value x danger at destination
        danger = (hit_en * (1 - hit_own)).reshape(-1, 1, 64) * P["danger"] + \
            (hit_en * hit_own).reshape(-1, 1, 64) * P["danger_def"]
        inter = ownV.reshape(-1, 64, 1) * danger
        # kernel leaving next ring when shrink comes
        kfw = m["ownK"].reshape(-1, 64, 1) * (N1 * u1).reshape(-1, 1, 64) * (-P["k_fw"])
        logits = fr + to - inter + kfw
        return logits.reshape(-1, 4096)


DEFAULT_POL = dict(k_val=4.0, ek_val=6.0, esc=0.5, esc_def=0.15, fw_from=0.6, k_from=-0.3,
                   cap=0.9, cap_hang=0.3, near_enK=0.25, fw_to=1.5, center=0.08,
                   danger=0.45, danger_def=0.15, k_fw=3.0)


def n_features() -> int:
    net = HandNet(1)
    net.pp = DEFAULT_POL
    with torch.no_grad():
        f, _ = net.features(torch.zeros(1, 16, 8, 8))
    return f.shape[1]


def build(weights: dict | np.ndarray, bias: float, pol: dict | None = None) -> HandNet:
    nf = n_features()
    net = HandNet(nf)
    net.pp = dict(DEFAULT_POL if pol is None else pol)
    if isinstance(weights, dict):
        w = np.zeros(nf, np.float32)
        for k, v in weights.items():
            w[FEATURES.index(k)] = v
    else:
        w = np.asarray(weights, np.float32)
    with torch.no_grad():
        net.lin.weight[:] = torch.tensor(w).reshape(1, -1)
        net.lin.bias[:] = float(bias)
    return net.eval()


class TorchEval:
    """Python evaluator for SearchAgent prototyping."""

    def __init__(self, net):
        self.net = net.eval()

    def __call__(self, planes):
        with torch.no_grad():
            p, v = self.net(torch.from_numpy(np.ascontiguousarray(planes, dtype=np.float32)))
        return p.numpy(), v.numpy()


V1_WEIGHTS = {
    "own_P": 0.15, "own_D": 0.45, "own_X": 0.25, "own_T": 0.3, "own_I": 0.45,
    "en_P": -0.15, "en_D": -0.45, "en_X": -0.25, "en_T": -0.3, "en_I": -0.45,
    "own_hang_P": -0.04, "own_hang_D": -0.12, "own_hang_X": -0.07, "own_hang_T": -0.08, "own_hang_I": -0.12,
    "en_hang_P": 0.06, "en_hang_D": 0.18, "en_hang_X": 0.1, "en_hang_T": 0.12, "en_hang_I": 0.18,
    "en_att2_D": 0.05, "en_att2_I": 0.05,
    "win_now": 3.0, "enK_att": 0.2, "ownK_att": -0.3, "ownK_att_g0": -0.6, "ownK_att_g1": -0.3,
    "g_own": 0.12, "g_en": -0.12, "ownK_zone_hit": -0.05, "enK_zone_hit": 0.05,
    "mob_own": 0.01, "mob_en": -0.01,
    "ownK_N1_u": -0.3, "ownK_N1_u1": -0.8, "ownK_N1_u1_a1": -1.0,
    "enK_N1_u": 0.3, "enK_N1_u1_mag": 4.0, "enK_N1_u1_cyan": 0.6,
    "ownV_N1_u1": -0.1, "enV_N1_u1_mag": 0.12, "enV_N1_u1_cyan": 0.05,
    "ownK_depth": 0.1, "enK_depth": -0.1, "Kdepth_diff_bare": 1.5,
    "mag": 0.15,
}

if __name__ == "__main__":
    nf = n_features()
    print(nf, FEATURES)
