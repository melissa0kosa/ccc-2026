"""One-off generator: derive fast4.py (hand_v4 graph) from fast.py."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
s = (HERE / "fast.py").read_text()


def rep(old, new):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new)


rep('"""Lean export graph equivalent to hand.HandNet (same features, same policy), ~4x fewer ONNX nodes.',
    '"""hand_v4 graph: FastNet (v3) + two-action reach maps for both sides, g1_t2 / hang2 value features,\n'
    'guard-aware Kernel-attack prior, opening sharpening, early Kernel evacuation term.')
rep('S_COLS = S_COLS + ["kdiff"]\n', '''S_COLS = S_COLS + ["kdiff"]
EXTRA = ["ownK_g1_t2", "enK_g1_t2"] + [f"own_hang2_{t}" for t in "PDXTI"] + [f"en_hang2a2_{t}" for t in "PDXTI"]
S_COLS = S_COLS + ["ownK_g1_t2", "enK_g1_t2"] + [f"own_hang2_{t}" for t in "PDXTI"] + [f"en_hang2_{t}" for t in "PDXTI"]
FMAP.update({"ownK_g1_t2": ("ownK_g1_t2", "one"), "enK_g1_t2": ("enK_g1_t2", "one")})
for t in "PDXTI":
    FMAP[f"own_hang2_{t}"] = (f"own_hang2_{t}", "one")
    FMAP[f"en_hang2a2_{t}"] = (f"en_hang2_{t}", "a2")
POL4 = dict(k_d2=3.0, k_esc_thr=2.0, guard=1.5, tie=0.05, ek_guard=1.5, danger2=0.25, k_fw2=0.6, warm=0.5)


def all_names():
    if not hand.FEATURES:
        hand.n_features()
    return list(hand.FEATURES) + EXTRA
''')
rep("class FastNet(nn.Module):", "class FastNet4(nn.Module):")
i0 = s.index("        sel2 = np.zeros((32, 6, 1, 1), np.float32)")
i1 = s.index("        tb = np.array(")
s = s[:i0] + '''        sel2 = np.zeros((64, 12, 1, 1), np.float32); sh2 = np.zeros((64, 1, 3, 3), np.float32)
        comb2 = np.zeros((2, 192, 1, 1), np.float32)
        for sd in range(2):
            for i, pt in enumerate(SLIDERS):
                dirs, ks = CAP[pt]
                for d, (dr, dc) in enumerate(ALL8):
                    ch = (sd * 4 + i) * 8 + d
                    sel2[ch, sd * 6 + pt, 0, 0] = 1; sh2[ch, 0, 1 - dr, 1 - dc] = 1
                    for k in (1, 2, 3):
                        if ((dr, dc) in dirs and k in ks) or (pt == DAEMON and k == 1):
                            comb2[sd, (k - 1) * 64 + ch, 0, 0] = 1
        jk2 = np.zeros((2, 12, 5, 5), np.float32)
        for sd in range(2):
            for dr, dc in KNIGHT:
                jk2[sd, sd * 6 + 2, 2 - dr, 2 - dc] = 1
            for dr, dc in ALL8:
                jk2[sd, sd * 6 + 5, 2 - dr, 2 - dc] = 1
        self.register_buffer("w_sel2", torch.tensor(sel2)); self.register_buffer("w_sh2", torch.tensor(sh2))
        self.register_buffer("w_comb2", torch.tensor(comb2)); self.register_buffer("w_jk2", torch.tensor(jk2))
''' + s[i1:]
rep('''        self.use_v4 = any(pv.get(k, 0.0) != 0.0 for k in ("k_d2", "k_esc_thr", "guard", "tie"))\n''', '')
rep('''        pv = dict(hand.DEFAULT_POL); pv.update(pol or {})''',
    '''        pv = dict(hand.DEFAULT_POL); pv.update(POL4); pv.update(pol or {})''')
rep('''        names = hand.FEATURES if hand.FEATURES else (hand.n_features() and hand.FEATURES)''',
    '''        names = all_names()''')
rep('''        self.register_buffer("Ss", torch.tensor(Ss)); self.register_buffer("Sgw", torch.tensor(Sg * w[None]))''',
    '''        self.register_buffer("Ss", torch.tensor(Ss)); self.register_buffer("Sg", torch.tensor(Sg))
        self.register_buffer("wv", torch.tensor(w).reshape(-1, 1))
        self.names = names''')
rep('''    def forward(self, x):
        P = self.pv''', '''    def features(self, x):
        return self.forward(x, feats_only=True)

    def forward(self, x, feats_only: bool = False):
        P = self.pv''')
i0 = s.index("        noesc = torch.relu(1.0 - SP[:, 33:35])")
i1 = s.index("        v = torch.tanh(lin)")
s = s[:i0] + '''        noesc = torch.relu(1.0 - SP[:, 33:35])
        kdiff = Q[:, 10:11] - Q[:, 11:12]
        bare = torch.relu(1.0 - Q[:, 17:18])
        # second ray pass: squares each side can attack after one quiet move
        y0 = F.conv2d(MV, self.w_sel2)
        q1 = F.conv2d(y0, self.w_sh2, padding=1, groups=64)
        q2 = F.conv2d(q1 * empty, self.w_sh2, padding=1, groups=64)
        q3 = F.conv2d(q2 * empty, self.w_sh2, padding=1, groups=64)
        A2 = F.conv2d(torch.cat([q1, q2, q3], 1), self.w_comb2) + F.conv2d(MV, self.w_jk2, padding=2)  # [own, en]
        hit2 = torch.clamp(A2, 0.0, 1.0) * (1.0 - hit)          # reachable in 2 actions only
        H2 = torch.cat([own5 * (hit2[:, 1:2] * (1.0 - hit_own)), en5 * (hit2[:, 0:1] * (1.0 - hit_en))], 1).sum((2, 3))
        S = torch.cat([Q, SP, torch.clamp(t2, 0.0, 3.0), ka, ka * g0, ka * g1, torch.clamp(g, 0.0, 3.0),
                       win_now, win2, thr, thr * a1, g0, kdiff, g1 * t2c, H2], 1)
        G = torch.cat([base[:, 10, 0:1, 0], a2, u, u1, u2, mag, cy, u1 * a1, u1 * mag, u1 * cy, u * cy,
                       u2 * cy, noesc[:, 0:1] * u1, noesc[:, 1:2] * u1 * cy, nlock, bare], 1)
        feats = (S @ self.Ss) * (G @ self.Sg)
        if feats_only:
            return feats
        lin = (feats @ self.wv).reshape(-1) + self.bias
''' + s[i1:]
i0 = s.index("        ownV2 = base[:, 8:9]; enV2 = base[:, 9:10]")
i1 = s.index("        return logits.reshape(-1, 4096), v")
s = s[:i0] + '''        ownV2 = base[:, 8:9]
        g0e = g0[:, 1:2].reshape(-1, 1, 1, 1)
        enV2 = base[:, 2:3] + enK * (P["ek_guard"] + (P["ek_val"] - P["ek_guard"]) * g0e)
        N1u = N1 * u1m
        u2m = u2.reshape(-1, 1, 1, 1)
        thr_o = thr[:, 0:1].reshape(-1, 1, 1, 1)
        fr = (ownV2 * ((hit_en - D) * P["esc"] + D * P["esc_def"] + N1u * P["fw_from"])
              + ownK * (P["k_from"] + thr_o * P["k_esc_thr"] + N1 * u2m * P["k_fw2"]) + self.tieb * P["tie"])
        to = (enV2 * (P["cap"] + P["cap_hang"] * (1.0 - hit_en)) + adj[:, 1:2] * empty * P["near_enK"]
              - N1u * P["fw_to"] + self.ring_idx * P["center"] + self.tieb2 * P["tie"])
        danger = (hit_en - D) * P["danger"] + D * P["danger_def"] + hit2[:, 1:2] * (1.0 - hit_own) * P["danger2"]
        ones = base[:, 10:11]
        gat = F.conv2d(base[:, 3:4], self.w_adj1, padding=1)       # own non-kernel pieces adjacent to t
        g0t = torch.relu(1.0 - gat)
        g1t = torch.relu(1.0 - torch.abs(gat - 1.0))
        kd = g0t * torch.clamp(A[:, 1:2] + A2[:, 1:2], 0.0, 1.0) + g1t * hit_en
        Lp = torch.cat([fr, ones, ownV2, ownK, base[:, 3:4]], 1).reshape(-1, 5, 64)
        Rp = torch.cat([ones, to, -danger, -N1u * P["k_fw"] - kd * P["k_d2"],
                        adj[:, 0:1] * empty * thr_o * P["guard"]], 1).reshape(-1, 5, 64)
        warm = torch.clamp((0.375 - u) * 8.0, 0.0, 1.0) * (1.0 - torch.clamp(nlock, 0.0, 1.0))
        logits = torch.matmul(Lp.transpose(1, 2), Rp) * (1.0 + P["warm"] * warm).reshape(-1, 1, 1)
''' + s[i1:]
s = s[:s.index("def check(")] + '''def neutral_check(pkl, wjson, n=3000):
    """With the new params neutral and new feature weights 0, FastNet4 must equal FastNet (v3)."""
    import json, fit, fast
    d = json.loads(Path(wjson).read_text())
    X, Z, IDX, CNT = fit.load([pkl]); X = torch.from_numpy(X[:n])
    neutral = dict(d.get("pol") or {})
    neutral.update(k_d2=0, k_esc_thr=0, guard=0, tie=0, danger2=0, k_fw2=0, warm=0,
                   ek_guard=neutral.get("ek_val", hand.DEFAULT_POL["ek_val"]))
    a = fast.FastNet(d["weights"], d["bias"], d.get("pol")).eval()
    b = FastNet4(d["weights"], d["bias"], neutral).eval()
    with torch.no_grad():
        p1, v1 = a(X); p2, v2 = b(X)
    mx = 0.0
    for i in range(n):
        ii = IDX[i].astype(np.int64); u_ = p1[i, ii]; w_ = p2[i, ii]
        mx = max(mx, ((u_ - u_.mean()) - (w_ - w_.mean())).abs().max().item())
    print(f"neutral check: max |dv| {(v1 - v2).abs().max().item():.2e}, policy {mx:.2e}")


if __name__ == "__main__":
    neutral_check(sys.argv[1], sys.argv[2])
'''
(HERE / "fast4.py").write_text(s)
print("wrote fast4.py")
