# COPY of agents/hand/fast.py (snapshot for az residual; only change: optional return_parts).
"""Lean export graph equivalent to hand.HandNet (same features, same policy), ~4x fewer ONNX nodes.

FastNet(weights_by_name, bias, pol) -> (policy (B,4096), value (B,)).
Verify with: uv run python levels/level4/agents/hand/fast.py data/sp_v1.pkl
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hand  # noqa: E402

RING = hand._ring_masks()                      # (4,8,8)
RING_OF = RING.argmax(0).reshape(64)           # ring index per square
RING_SIZE = RING.reshape(4, 64).sum(1)         # 28,20,12,4
VALS = [1.0, 3.0, 1.7, 2.0, 3.0]

# ---- column names
Q_COLS = ["own_P", "own_D", "own_X", "own_T", "own_I", "en_P", "en_D", "en_X", "en_T", "en_I",
          "ownK_depth", "enK_depth", "mag", "a2", "u", "nlock", "matdiff", "nk_count"]
SP_COLS = ([f"own_att_{t}" for t in "PDXTI"] + [f"en_att_{t}" for t in "PDXTI"]
           + [f"own_hang_{t}" for t in "PDXTI"] + [f"en_hang_{t}" for t in "PDXTI"])
LR_COLS = ["ownK_att_n", "enK_att_n", "ownK_zone_hit", "enK_zone_hit", "g_own_raw", "g_en_raw",
           "mob_own", "mob_en", "okN1", "ekN1", "okN2", "ovN1", "evN1", "esc_own", "esc_en"]
NL_COLS = ["t2_own3", "t2_en3", "ownK_att", "enK_att", "ownK_att_g0", "enK_att_g0",
           "ownK_att_g1", "enK_att_g1", "g_own", "g_en", "win_now", "win2",
           "ownK_thr", "enK_thr", "ownK_thr_a1", "enK_thr_a1", "ownK_g0", "enK_g0"]
S_COLS = Q_COLS + SP_COLS + LR_COLS + NL_COLS
G_COLS = ["one", "a2", "u", "u1", "u2", "mag", "cy", "u1a1", "u1mag", "u1cy", "ucy", "u2cy",
          "noesc_own_u1", "noesc_en_u1cy", "nlock", "bare"]

# feature name -> (S column, G column)
FMAP = {}
for t in "PDXTI":
    for k in ("own_", "en_", "own_att_", "own_hang_", "en_att_", "en_hang_"):
        FMAP[f"{k}{t}"] = (f"{k}{t}", "one")
    FMAP[f"en_att2_{t}"] = (f"en_att_{t}", "a2")
for n in ["win_now", "enK_att", "ownK_att", "g_own", "g_en", "win2", "ownK_thr", "enK_thr",
          "ownK_thr_a1", "enK_thr_a1", "ownK_g0", "enK_g0", "ownK_att_n", "enK_att_n",
          "ownK_zone_hit", "enK_zone_hit", "mob_own", "mob_en", "ownK_depth", "enK_depth"]:
    FMAP[n] = (n, "one")
FMAP.update({
    "enK_att_g0": ("enK_att_g0", "one"), "enK_att_g1": ("enK_att_g1", "one"),
    "ownK_att_g0": ("ownK_att_g0", "one"), "ownK_att_g1": ("ownK_att_g1", "one"),
    "t2_en": ("t2_en3", "one"), "t2_own": ("t2_own3", "one"),
    "ownK_N1_u": ("okN1", "u"), "ownK_N1_u1": ("okN1", "u1"), "ownK_N1_u1_a1": ("okN1", "u1a1"),
    "ownK_N2_u1": ("okN2", "u1"), "enK_N1_u": ("ekN1", "u"), "enK_N1_u1_mag": ("ekN1", "u1mag"),
    "enK_N1_u1_cyan": ("ekN1", "u1cy"), "ownK_N1_u1_mag": ("okN1", "u1mag"),
    "ownK_N1_u2": ("okN1", "u2"), "ownK_N1_u_cyan": ("okN1", "ucy"),
    "ownK_N1_u2_cyan": ("okN1", "u2cy"), "ownK_N1_u1_cyan": ("okN1", "u1cy"),
    "ownV_N1_u2_cyan": ("ovN1", "u2cy"), "ownK_N1_u1_noesc": ("okN1", "noesc_own_u1"),
    "enK_N1_u2": ("ekN1", "u2"), "enK_N1_u1_noesc_cyan": ("ekN1", "noesc_en_u1cy"),
    "ownV_N1_u1": ("ovN1", "u1"), "enV_N1_u1_mag": ("evN1", "u1mag"),
    "enV_N1_u1_cyan": ("evN1", "u1cy"), "ownV_N1_u": ("ovN1", "u"), "enV_N1_u": ("evN1", "u"),
    "Kdepth_diff_bare": ("kdiff", "bare"), "mag": ("mag", "one"), "a2": ("a2", "one"),
    "mag_late": ("mag", "u1"), "nlock": ("nlock", "one"), "nlock_mag": ("nlock", "mag"),
    "mat_x_nlock": ("matdiff", "nlock"),
})
S_COLS = S_COLS + ["kdiff"]


class FastNet(nn.Module):
    def __init__(self, weights: dict, bias: float, pol: dict):
        super().__init__()
        hn = hand.HandNet(1)
        for k in ("w_sel", "w_shift", "w_comb", "w_jk"):
            self.register_buffer(k, getattr(hn, k).clone())
        # reorg 1x1 conv over cat[sl(24), jk(6)] -> 28 maps
        R = np.zeros((28, 30), np.float32)
        for c in (0, 1, 2, 3):
            R[0, c] = 1; R[1, 4 + c] = 1
        R[0, 24] = R[0, 25] = 1; R[1, 26] = R[1, 27] = 1
        mv_own = [8, 9, 24, 10, 11, 25]; mv_en = [12, 13, 26, 14, 15, 27]
        rev_en = [20, 21, 29, 22, 23, 27]; rev_own = [16, 17, 28, 18, 19, 25]
        for i in range(6):
            R[2 + i, mv_own[i]] = 1; R[8 + i, mv_en[i]] = 1
            R[14 + i, rev_en[i]] = 1; R[20 + i, rev_own[i]] = 1
        R[26, 25] = 1; R[27, 27] = 1
        self.register_buffer("w_reorg", torch.tensor(R).reshape(28, 30, 1, 1))
        self.register_buffer("w_adj1", hn.w_adj.clone())
        pv = dict(hand.DEFAULT_POL); pv.update(pol or {})
        # base 1x1 conv: 0 empty, 1 ownV, 2 enV, 3 ownNK, 4 enNK, 5 ownK, 6 enK, 7 notlocked,
        # 8 ownV2 (policy value incl kernel), 9 enV2, 10 ones
        Bw = np.zeros((11, 16), np.float32); Bb = np.zeros(11, np.float32)
        Bw[0, :13] = -1; Bb[0] = 1
        for t in range(5):
            Bw[1, t] = VALS[t]; Bw[2, 6 + t] = VALS[t]; Bw[3, t] = 1; Bw[4, 6 + t] = 1
            Bw[8, t] = VALS[t]; Bw[9, 6 + t] = VALS[t]
        Bw[5, 5] = 1; Bw[6, 11] = 1; Bw[7, 12] = -1; Bb[7] = 1
        Bw[8, 5] = pv["k_val"]; Bw[9, 11] = pv["ek_val"]; Bb[10] = 1
        self.register_buffer("w_base", torch.tensor(Bw).reshape(11, 16, 1, 1))
        self.register_buffer("b_base", torch.tensor(Bb))
        # Q gemm over flatten(x) (1024 = 16*64)
        nq = len(Q_COLS)
        Wq = np.zeros((16, 64, nq), np.float32); bq = np.zeros(nq, np.float32)
        lrw = np.zeros((64, 4), np.float32)  # locked plane -> lr_r
        for sq in range(64):
            lrw[sq, RING_OF[sq]] = 1.0 / RING_SIZE[RING_OF[sq]]
        for t in range(5):
            Wq[t, :, t] = 1; Wq[6 + t, :, 5 + t] = 1
            Wq[t, :, 16] = VALS[t]; Wq[6 + t, :, 16] = -VALS[t]
            Wq[t, :, 17] = 1; Wq[6 + t, :, 17] = 1
        Wq[5, :, 10] = RING_OF; Wq[11, :, 11] = RING_OF
        Wq[12, :, 10] -= lrw.sum(1); Wq[12, :, 11] -= lrw.sum(1); Wq[12, :, 15] = lrw.sum(1)
        Wq[15, 0, 12] = 1; Wq[13, 0, 13] = 2; bq[13] = -1; Wq[14, 0, 14] = 1
        self.register_buffer("Wq", torch.tensor(Wq.reshape(1024, nq)))
        self.register_buffer("bq", torch.tensor(bq))
        # N1/N2 maps: linear in locked plane
        WN = np.zeros((1024, 128), np.float32); bN = np.zeros(128, np.float32)
        lock_rows = slice(12 * 64, 13 * 64)
        def lr_vec(r):  # (64,) weights giving lr_r from locked plane
            v = np.zeros(64, np.float32); v[RING_OF == r] = 1.0 / RING_SIZE[r]; return v
        for sq in range(64):
            r = RING_OF[sq]
            # N1 = lr_{r-1} - lr_r
            WN[lock_rows, sq] -= lr_vec(r)
            if r == 0:
                bN[sq] = 1
            else:
                WN[lock_rows, sq] += lr_vec(r - 1)
            # N2 = N1 of ring r-1
            if r >= 1:
                WN[lock_rows, 64 + sq] -= lr_vec(r - 1)
                if r - 1 == 0:
                    bN[64 + sq] = 1
                else:
                    WN[lock_rows, 64 + sq] += lr_vec(r - 2)
        self.register_buffer("WN", torch.tensor(WN)); self.register_buffer("bN", torch.tensor(bN))
        self.register_buffer("ring_idx", torch.tensor(RING_OF.astype(np.float32)).reshape(1, 1, 8, 8))
        # second ray pass: enemy attack after one quiet move (from mv_en maps, 6 types)
        from hand import SLIDERS, ALL8, CAP, DAEMON, KNIGHT
        sel2 = np.zeros((32, 6, 1, 1), np.float32); sh2 = np.zeros((32, 1, 3, 3), np.float32)
        comb2 = np.zeros((1, 96, 1, 1), np.float32)
        for i, pt in enumerate(SLIDERS):
            dirs, ks = CAP[pt]
            for d, (dr, dc) in enumerate(ALL8):
                ch = i * 8 + d
                sel2[ch, pt, 0, 0] = 1; sh2[ch, 0, 1 - dr, 1 - dc] = 1
                for k in (1, 2, 3):
                    if ((dr, dc) in dirs and k in ks) or (pt == DAEMON and k == 1):
                        comb2[0, (k - 1) * 32 + ch, 0, 0] = 1
        jk2 = np.zeros((1, 6, 5, 5), np.float32)
        for dr, dc in KNIGHT:
            jk2[0, 2, 2 - dr, 2 - dc] = 1
        for dr, dc in ALL8:
            jk2[0, 5, 2 - dr, 2 - dc] = 1
        self.register_buffer("w_sel2", torch.tensor(sel2)); self.register_buffer("w_sh2", torch.tensor(sh2))
        self.register_buffer("w_comb2", torch.tensor(comb2)); self.register_buffer("w_jk2", torch.tensor(jk2))
        tb = np.array([((sq * 37) % 64) / 64.0 for sq in range(64)], np.float32).reshape(1, 1, 8, 8)
        self.register_buffer("tieb", torch.tensor(tb))
        self.register_buffer("tieb2", torch.tensor(np.ascontiguousarray(tb[..., ::-1])))
        self.use_v4 = any(pv.get(k, 0.0) != 0.0 for k in ("k_d2", "k_esc_thr", "guard", "tie"))
        # t2 group sum (12 -> 2): [t2_own (enemy moves vs my K), t2_en]
        Gt = np.zeros((12, 2), np.float32); Gt[6:, 0] = 1; Gt[:6, 1] = 1
        self.register_buffer("Gt", torch.tensor(Gt))
        # feature selection
        names = hand.FEATURES if hand.FEATURES else (hand.n_features() and hand.FEATURES)
        nf = len(names)
        Ss = np.zeros((len(S_COLS), nf), np.float32); Sg = np.zeros((len(G_COLS), nf), np.float32)
        w = np.zeros(nf, np.float32)
        for i, n in enumerate(names):
            s, g = FMAP[n]
            Ss[S_COLS.index(s), i] = 1; Sg[G_COLS.index(g), i] = 1
            w[i] = weights.get(n, 0.0)
        # fold weights into the gate selection: lin = sum((S@Ss) * (G@(Sg*w)))
        self.register_buffer("Ss", torch.tensor(Ss)); self.register_buffer("Sgw", torch.tensor(Sg * w[None]))
        self.bias = float(bias)
        self.pv = pv

    def forward(self, x):
        P = self.pv
        base = F.conv2d(x, self.w_base, self.b_base)      # (B,11,8,8)
        empty = base[:, 0:1]
        x0 = F.conv2d(x, self.w_sel)
        r1 = F.conv2d(x0, self.w_shift, padding=1, groups=80)
        r2 = F.conv2d(r1 * empty, self.w_shift, padding=1, groups=80)
        r3 = F.conv2d(r2 * empty, self.w_shift, padding=1, groups=80)
        sl = F.conv2d(torch.cat([r1, r2, r3], 1), self.w_comb)
        jk = F.conv2d(x, self.w_jk, padding=2)
        R = F.conv2d(torch.cat([sl, jk], 1), self.w_reorg)   # (B,28,8,8)
        A = R[:, 0:2]                                         # A_own, A_en
        hit = torch.clamp(A, 0.0, 1.0)
        hit_own = hit[:, 0:1]; hit_en = hit[:, 1:2]
        D = hit_own * hit_en
        adj = R[:, 26:28]                                      # adj_ownK, adj_enK
        xf = x.flatten(1)
        Q = xf @ self.Wq + self.bq                             # (B,18)
        NN = (xf @ self.WN + self.bN).reshape(-1, 2, 8, 8)     # N1, N2
        N1 = NN[:, 0:1]
        own5 = x[:, 0:5]; en5 = x[:, 6:11]
        ownK = base[:, 5:6]; enK = base[:, 6:7]
        # products
        Patt = torch.cat([own5 * hit_en, en5 * hit_own], 1)
        Phang = torch.cat([own5 * (hit_en - D), en5 * (hit_own - D)], 1)
        nl = base[:, 7:8]
        Lm = torch.cat([ownK, enK, adj, adj, hit, NN[:, 0:1], NN[:, 0:1], NN[:, 1:2],
                        NN[:, 0:1], NN[:, 0:1], adj * empty], 1)
        Rm = torch.cat([A[:, 1:2], A[:, 0:1], hit_en, hit_own, base[:, 3:5], nl, nl, ownK, enK, ownK,
                        base[:, 1:3], 1.0 - N1, 1.0 - N1], 1)
        MV = R[:, 2:14] * empty
        T12 = (MV * R[:, 14:26]).sum((2, 3))
        SP = torch.cat([Patt, Phang, Lm * Rm], 1).sum((2, 3))  # (B,20+15)
        t2 = T12 @ self.Gt                                      # [t2_own, t2_en]
        # scalars
        mag = Q[:, 12:13]; a2 = Q[:, 13:14]; u = Q[:, 14:15]; nlock = Q[:, 15:16]
        a1 = 1.0 - a2; cy = 1.0 - mag
        u1 = torch.relu(u - 0.875) * 8.0
        u2 = torch.clamp((u - 0.625) * 8.0, 0.0, 1.0)
        ka = torch.clamp(SP[:, 20:22], 0.0, 1.0)               # [own attacked, en attacked]
        g = SP[:, 24:26]                                        # [g_own, g_en]
        g0 = torch.relu(1.0 - g)
        g1 = torch.relu(1.0 - torch.abs(g - 1.0))
        ka_en = ka[:, 1:2]
        win_now = torch.clamp(ka_en * (g0[:, 1:2] + g1[:, 1:2] * a2), 0.0, 1.0)
        t2c = torch.clamp(t2, 0.0, 1.0)
        win2 = torch.clamp(win_now + a2 * g0[:, 1:2] * t2c[:, 1:2], 0.0, 1.0)
        thr = torch.clamp(g0 * torch.clamp(ka + t2c, 0.0, 1.0) + g1 * ka, 0.0, 1.0)
        noesc = torch.relu(1.0 - SP[:, 33:35])
        kdiff = Q[:, 10:11] - Q[:, 11:12]
        bare = torch.relu(1.0 - Q[:, 17:18])
        S = torch.cat([Q, SP, torch.clamp(t2, 0.0, 3.0), ka, ka * g0, ka * g1, torch.clamp(g, 0.0, 3.0),
                       win_now, win2, thr, thr * a1, g0, kdiff], 1)
        G = torch.cat([base[:, 10, 0:1, 0], a2, u, u1, u2, mag, cy, u1 * a1, u1 * mag, u1 * cy, u * cy,
                       u2 * cy, noesc[:, 0:1] * u1, noesc[:, 1:2] * u1 * cy, nlock, bare], 1)
        lin = ((S @ self.Ss) * (G @ self.Sgw)).sum(1) + self.bias
        v = torch.tanh(lin)
        # ---------------- policy: logits[f,t] = sum_k Lp[k,f] * Rp[k,t]
        u1m = u1.reshape(-1, 1, 1, 1)
        ownV2 = base[:, 8:9]; enV2 = base[:, 9:10]
        N1u = N1 * u1m
        fr = ownV2 * ((hit_en - D) * P["esc"] + D * P["esc_def"] + N1u * P["fw_from"]) + ownK * P["k_from"]
        to = (enV2 * (P["cap"] + P["cap_hang"] * (1.0 - hit_en)) + adj[:, 1:2] * empty * P["near_enK"]
              - N1u * P["fw_to"] + self.ring_idx * P["center"])
        danger = (hit_en - D) * P["danger"] + D * P["danger_def"]
        ones = base[:, 10:11]
        if not self.use_v4:
            Lp = torch.cat([fr, ones, ownV2, ownK], 1).reshape(-1, 4, 64)
            Rp = torch.cat([ones, to, -danger, -N1u * P["k_fw"]], 1).reshape(-1, 4, 64)
        else:
            mv_en = MV[:, 6:12]
            y0 = F.conv2d(mv_en, self.w_sel2)
            q1 = F.conv2d(y0, self.w_sh2, padding=1, groups=32)
            q2 = F.conv2d(q1 * empty, self.w_sh2, padding=1, groups=32)
            q3 = F.conv2d(q2 * empty, self.w_sh2, padding=1, groups=32)
            A2 = F.conv2d(torch.cat([q1, q2, q3], 1), self.w_comb2) + F.conv2d(mv_en, self.w_jk2, padding=2)
            gat = F.conv2d(base[:, 3:4], self.w_adj1, padding=1)       # own non-kernel pieces adjacent to t
            g0t = torch.relu(1.0 - gat)
            g1t = torch.relu(1.0 - torch.abs(gat - 1.0))
            kd = g0t * torch.clamp(A[:, 1:2] + A2, 0.0, 1.0) + g1t * hit_en
            thr_o = thr[:, 0:1].reshape(-1, 1, 1, 1)
            fr = fr + ownK * thr_o * P["k_esc_thr"] + self.tieb * P["tie"]
            to = to + self.tieb2 * P["tie"]
            Lp = torch.cat([fr, ones, ownV2, ownK, base[:, 3:4]], 1).reshape(-1, 5, 64)
            Rp = torch.cat([ones, to, -danger, -N1u * P["k_fw"] - kd * P["k_d2"],
                            adj[:, 0:1] * empty * thr_o * P["guard"]], 1).reshape(-1, 5, 64)
        logits = torch.matmul(Lp.transpose(1, 2), Rp)
        if getattr(self, "return_parts", False):  # az residual: raw value pre-activation + feature vectors
            return logits.reshape(-1, 4096), lin, S, G
        return logits.reshape(-1, 4096), v


def check(pkl, weights, bias, pol, n=3000):
    import fit
    X, Z, IDX, CNT = fit.load([pkl])
    X = X[:n]
    ref = hand.build(weights, bias, pol)
    fn = FastNet(weights, bias, pol).eval()
    with torch.no_grad():
        xt = torch.from_numpy(X)
        p1, v1 = ref(xt)
        p2, v2 = fn(xt)
    dv = (v1 - v2).abs().max().item()
    # compare policy on legal moves only, after per-row softmax shift
    mx = 0.0
    for i in range(len(X)):
        a = p1[i, IDX[i].astype(np.int64)]; b = p2[i, IDX[i].astype(np.int64)]
        mx = max(mx, ((a - a.mean()) - (b - b.mean())).abs().max().item())
    print(f"max |dv| {dv:.2e}, max policy diff (legal, centered) {mx:.2e}")
    return dv, mx


if __name__ == "__main__":
    import json
    pkl = sys.argv[1]
    wj = sys.argv[2] if len(sys.argv) > 2 else None
    if wj:
        d = json.loads(Path(wj).read_text()); check(pkl, d["weights"], d.get("bias", 0.0), d.get("pol"))
    else:
        check(pkl, hand.V1_WEIGHTS, 0.0, None)
