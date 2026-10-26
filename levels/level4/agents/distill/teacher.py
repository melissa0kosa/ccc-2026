"""Alpha-beta teacher for COLONY: ONE (2 actions per turn), uses the engine Board API only."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # levels/level4
sys.path.insert(0, str(ROOT))
from c1 import *  # noqa: F401,E402
from colony_one import constants, encoding  # noqa: E402
from colony_one.pieces import PieceType  # noqa: E402
from colony_one.rules import legal_actions_for_team  # noqa: E402

KERNEL = PieceType.KERNEL
VAL = {PieceType.PING: 1.0, PieceType.EXPLOIT: 1.7, PieceType.TRACE: 2.0,
       PieceType.DAEMON: 3.0, PieceType.ICE: 3.3, PieceType.KERNEL: 0.0}
ORDER_VAL = dict(VAL)
ORDER_VAL[KERNEL] = 50.0
WIN = 100.0
SHRINK = set(constants.SHRINK_ROUNDS)
MAGENTA_ = constants.MAGENTA
CYAN_ = constants.CYAN


def opp(t):
    return MAGENTA_ if t == CYAN_ else CYAN_


def ring_of(r, c):
    return int(max(abs(r - 3.5), abs(c - 3.5)))  # 0..3


def next_lock_ring(round_number):
    """(ring index that locks next (3=outer), rounds until it locks incl. current)"""
    for i, s in enumerate(constants.SHRINK_ROUNDS):
        if s >= round_number:
            return 3 - i, s - round_number
    return None, None


RING = [[int(max(abs(r - 3.5), abs(c - 3.5))) for c in range(8)] for r in range(8)]
KPEN = {0: 5.0, 1: 2.5, 2: 1.0, 3: 0.4}
PPEN = {0: 0.5, 1: 0.25}
_PINFO = {}


def _pinfo(p):
    pt = p.piece_type
    return (pt is KERNEL, VAL[pt], p.team)


NB = [[[(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
        if (dr or dc) and 0 <= r + dr < 8 and 0 <= c + dc < 8] for c in range(8)] for r in range(8)]
_NLR = {}


def static_eval(board, team, round_number):
    """Score from `team`'s perspective (pawn units)."""
    nl = _NLR.get(round_number)
    if nl is None:
        lr, rem = next_lock_ring(round_number)
        if lr is None:
            lr, rem = 99, 99
        nl = _NLR[round_number] = (lr, KPEN.get(rem, 0.15), PPEN.get(rem, 0.08))
    lr, kpen, ppen = nl
    s = 0.0
    kpos = []
    grid = board.grid
    info = _PINFO
    for r in range(8):
        row = grid[r]
        RR = RING[r]
        for c in range(8):
            p = row[c]
            if p is None:
                continue
            k = id(p)
            inf = info.get(k)
            if inf is None:
                inf = info[k] = _pinfo(p)
            isk, v, t = inf
            if isk:
                kpos.append((t, r, c))
                continue
            rg = RR[c]
            x = v - 0.04 * rg
            if rg >= lr:
                x -= v * ppen
            if t == team:
                s += x
            else:
                s -= x
    for t, r, c in kpos:
        guards = 0
        for rr, cc in NB[r][c]:
            q = grid[rr][cc]
            if q is not None and q.team == t and not info[id(q)][0]:
                guards += 1
        g = 0.6 * min(guards, 1) + 0.3 * min(max(guards - 1, 0), 1) + 0.1 * min(max(guards - 2, 0), 2)
        rg = RING[r][c]
        k = g - 0.15 * rg
        if rg >= lr:
            k -= kpen
        if t == team:
            s += k
        else:
            s -= k
    return s


def ordered(board, team, acts):
    """Captures first (by victim value, kernel highest)."""
    grid = board.grid
    caps, quiet = [], []
    for a in acts:
        tr, tc = a.to_pos
        q = grid[tr][tc]
        if q is not None and q.team != team:
            caps.append((ORDER_VAL[q.piece_type], a))
        else:
            quiet.append(a)
    caps.sort(key=lambda x: -x[0])
    return [a for _, a in caps] + quiet, len(caps)


class Searcher:
    def __init__(self, qdepth=2):
        self.nodes = 0
        self.qdepth = qdepth
        self.tt = {}

    # returns value from `team`'s perspective; handles one action and its consequences
    def child_value(self, board, team, al, rnd, a, depth, alpha, beta, quies):
        rec = board.apply_action(a)
        try:
            cp = rec.captured_piece
            if cp is not None and cp.piece_type is KERNEL:
                return WIN + depth  # prefer faster wins
            if al > 1:
                if quies:
                    return self.qsearch(board, team, al - 1, rnd, alpha, beta)
                return self.negamax(board, team, al - 1, rnd, depth - 1, alpha, beta)
            # turn end
            if board.only_kernels_remain():
                w = board.kernel_endgame_winner(acting_team=team)
                return WIN if w == team else -WIN
            fw = None
            if team == MAGENTA_ and rnd in SHRINK:
                fw = board.apply_firewall_shrink()
            try:
                if fw is not None:
                    kk = [p for _, _, p in fw.killed if p.piece_type is KERNEL]
                    if kk:
                        w = MAGENTA_ if len(kk) == 2 else opp(kk[0].team)
                        return WIN if w == team else -WIN
                nt = opp(team)
                nr = rnd + 1 if team == MAGENTA_ else rnd
                if quies:
                    return -static_eval(board, nt, nr)
                return -self.negamax(board, nt, 2, nr, depth - 1, -beta, -alpha)
            finally:
                if fw is not None:
                    board.undo_firewall_shrink(fw)
        finally:
            board.undo_action(rec)

    def qsearch(self, board, team, al, rnd, alpha, beta):
        """Remaining actions of the current turn, captures only, stand pat allowed."""
        self.nodes += 1
        stand = static_eval(board, team, rnd)
        if stand >= beta:
            return stand
        best = stand
        if stand > alpha:
            alpha = stand
        acts = legal_actions_for_team(board, team)
        acts, ncap = ordered(board, team, acts)
        for a in acts[:ncap]:
            v = self.child_value(board, team, al, rnd, a, 0, alpha, beta, True)
            if v > best:
                best = v
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        return best

    def negamax(self, board, team, al, rnd, depth, alpha, beta):
        if depth <= 0:
            return self.qsearch(board, team, al, rnd, alpha, beta)
        self.nodes += 1
        key = (tuple([id(p) for row in board.grid for p in row]), team, al, rnd)
        e = self.tt.get(key)
        bm = None
        if e is not None:
            d, v, flag, bm = e
            if d >= depth:
                if flag == 0:
                    return v
                if flag == 1 and v >= beta:
                    return v
                if flag == 2 and v <= alpha:
                    return v
        alpha0 = alpha
        acts = legal_actions_for_team(board, team)
        if not acts:
            return static_eval(board, team, rnd)
        acts, _ = ordered(board, team, acts)
        if bm is not None:
            try:
                acts.remove(bm)
                acts.insert(0, bm)
            except ValueError:
                pass
        best = -1e9
        bact = None
        for a in acts:
            v = self.child_value(board, team, al, rnd, a, depth, alpha, beta, False)
            if v > best:
                best = v
                bact = a
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        flag = 2 if best <= alpha0 else (1 if best >= beta else 0)
        self.tt[key] = (depth, best, flag, bact)
        return best

    def root(self, board, team, al, rnd, depth=None, margin=3.0):
        """Scores for every legal action (moves worse than best-margin get an upper bound)."""
        self.tt = {}
        if depth is None:
            depth = al + 2  # rest of our turn + full opponent turn
        acts = legal_actions_for_team(board, team)
        acts, _ = ordered(board, team, acts)
        scores = []
        best = -1e9
        for a in acts:
            alpha = best - margin if best > -1e8 else -1e9
            v = self.child_value(board, team, al, rnd, a, depth, alpha, 1e9, False)
            scores.append(v)
            if v > best:
                best = v
        return acts, scores, best


if __name__ == "__main__":
    import time
    import numpy as np
    env = ColonyOneEnv()
    rng = np.random.default_rng(0)
    S = Searcher()
    tt = []
    while not env.terminated:
        t = time.perf_counter()
        S.nodes = 0
        acts, sc, best = S.root(env.board, env.current_team, env.actions_left, env.round_number)
        dt = time.perf_counter() - t
        tt.append(dt)
        i = int(np.argmax(np.array(sc) + rng.random(len(sc)) * 0.01))
        print(f"a{env.action} {env.current_team[:1]} al{env.actions_left} r{env.round_number} "
              f"n={len(acts)} nodes={S.nodes} {dt:.2f}s best={best:.2f} play {acts[i]}", flush=True)
        env.step(acts[i])
    print("winner", env.winner, "mean t", np.mean(tt), "max", np.max(tt))
