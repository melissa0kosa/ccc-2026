"""Verify az_res4f_v2_eg.onnx: (1) bit parity with the base outside the gate on real-game positions,
(2) brute force vs the env on generated gated positions (value and veto), incl. the Magenta+urgency gate-off.
usage: eg_verify.py base.onnx eg.onnx [n_random]"""
import pickle, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import numpy as np
import onnxruntime as ort
from c1 import ColonyOneEnv, encoding, CYAN, MAGENTA
from colony_one.pieces import Piece, PieceType
from colony_one import constants


def sess(p):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])


def run(s, X):
    P, V = [], []
    for i in range(0, len(X), 16):
        p, v = s.run(None, {"observation": X[i:i + 16]})
        P.append(p); V.append(v)
    return np.concatenate(P), np.concatenate(V)


def parity(base, eg):
    X = []
    for f in ["sp_v4f.pkl", "sp_v3.pkl"]:
        for rec in pickle.load(open(HERE / "data" / f, "rb")):
            x = np.empty((16, 8, 8), np.float32); x[:13] = rec[0]; x[13:] = rec[1][:, None, None]; X.append(x)
    X = np.stack(X)
    idx = np.random.default_rng(0).choice(len(X), 6000, replace=False)
    X = X[idx]
    pb, vb = run(base, X); pe, ve = run(eg, X)
    own_nk = X[:, 0:5].sum((1, 2, 3)); en_nk = X[:, 6:11].sum((1, 2, 3))
    al1 = X[:, 13, 0, 0] < 0.75
    fwoff = (X[:, 14, 0, 0] > 0.999) & (X[:, 15, 0, 0] > 0.5)
    gated = (own_nk == 0) & (en_nk <= 1) & al1 & ~fwoff & (X[:, 5].sum((1, 2)) == 1) & (X[:, 11].sum((1, 2)) == 1)
    same_v = (vb == ve) | gated
    same_p = (pb == pe).all(1) | gated
    print(f"[parity] {len(X)} real-game positions: gated {gated.sum()}, value bit-identical outside gate: "
          f"{same_v.all()} ({(~(vb == ve) & ~gated).sum()} diffs), policy bit-identical outside gate: {same_p.all()}")
    return X


def gen_position(rng, case):
    env = ColonyOneEnv()
    b = env.board
    for r in range(8):
        for c in range(8):
            b.grid[r][c] = None
    nl = int(rng.integers(0, 3))
    for _ in range(nl):
        b.lock_outer_ring()
    free = [(r, c) for r in range(8) for c in range(8) if not b.is_locked(r, c)]
    team = CYAN if rng.random() < 0.5 else MAGENTA
    enemy = MAGENTA if team == CYAN else CYAN
    k = 3 if case == "A" else 2
    cells = [free[i] for i in rng.choice(len(free), k, replace=False)]
    # bias towards close Kernels so the interesting cases appear
    if rng.random() < 0.7:
        r0, c0 = cells[0]
        near = [(r, c) for (r, c) in free if max(abs(r - r0), abs(c - c0)) <= 3 and (r, c) != (r0, c0)]
        pick = [near[i] for i in rng.choice(len(near), min(k - 1, len(near)), replace=False)]
        if len(pick) == k - 1:
            cells = [cells[0]] + pick
    b.grid[cells[0][0]][cells[0][1]] = Piece(PieceType.KERNEL, team)
    b.grid[cells[1][0]][cells[1][1]] = Piece(PieceType.KERNEL, enemy)
    if case == "A":
        pt = [PieceType.PING, PieceType.DAEMON, PieceType.EXPLOIT, PieceType.TRACE, PieceType.ICE][int(rng.integers(5))]
        b.grid[cells[2][0]][cells[2][1]] = Piece(pt, enemy)
    lo = {0: 1, 1: 9, 2: 17}[nl]; hi = {0: 8, 1: 16, 2: 24}[nl]
    env.round_number = int(rng.integers(lo, hi + 1)) if rng.random() < 0.7 else hi   # often the shrink round
    env.current_team = team
    env.actions_left = 1
    return env


def brute(env):
    team = env.current_team
    res = {}
    for a in env.legal_actions():
        e = env.clone(); e.step(a)
        res[encoding.action_index(a, team)] = ("win" if e.winner == team else "loss") if e.terminated else "cont"
    return res


def main():
    base, eg = sess(sys.argv[1]), sess(sys.argv[2])
    parity(base, eg)
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    rng = np.random.default_rng(1)
    stats = dict(A_win=0, A_lose=0, B_win=0, B_lose=0, off=0, bad_v=0, bad_veto=0, bad_off=0, B_nonterm=0)
    examples = []
    for i in range(n):
        case = "A" if i % 2 == 0 else "B"
        env = gen_position(rng, case)
        x = env.observation()[None]
        pb, vb = base.run(None, {"observation": x}); pe, ve = eg.run(None, {"observation": x})
        team = env.current_team
        u1 = encoding.firewall_urgency(env.round_number) == 1.0
        res = brute(env)
        if team == MAGENTA and u1:
            stats["off"] += 1
            if not (vb[0] == ve[0] and (pb == pe).all()):
                stats["bad_off"] += 1
            continue
        any_win = any(v == "win" for v in res.values())
        if case == "A":
            expect = 1.0 if any_win else -0.95
        else:
            expect = 1.0 if any_win else -1.0
            if any(v == "cont" for v in res.values()):
                stats["B_nonterm"] += 1
        stats[f"{case}_{'win' if any_win else 'lose'}"] += 1
        if abs(float(ve[0]) - expect) > 1e-6:
            stats["bad_v"] += 1
            if len(examples) < 5:
                examples.append((case, team, env.round_number, float(ve[0]), expect, env.render()))
        d = (pe[0] - pb[0])
        for a, out in res.items():
            vetoed = d[a] < -1.0
            should = case == "A" and out == "loss"
            if vetoed != should or (not vetoed and d[a] != 0.0):
                stats["bad_veto"] += 1
                break
    print("[brute force]", stats)
    for ex in examples:
        print(ex[:5]); print(ex[5])


if __name__ == "__main__":
    main()
