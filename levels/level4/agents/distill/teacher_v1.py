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


def static_eval(board, team, round_number):
    """Score from `team`'s perspective (pawn units)."""
    s = 0.0
    kpos = {}
    grid = board.grid
    lr, rem = next_lock_ring(round_number)
    for r in range(8):
        row = grid[r]
        for c in range(8):
            p = row[c]
            if p is None:
                continue
            sg = 1.0 if p.team == team else -1.0
            pt = p.piece_type
            if pt is KERNEL:
                kpos[p.team] = (r, c)
                continue
            v = VAL[pt]
            rg = ring_of(r, c)
            pen = 0.0
            if lr is not None and rg >= lr:
                # piece will burn when the ring locks
                pen = v * (0.5 if rem == 0 else 0.25 if rem == 1 else 0.08)
            s += sg * (v - pen - 0.04 * rg)
    for t, (r, c) in kpos.items():
        sg = 1.0 if t == team else -1.0
        guards = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < 8 and 0 <= cc < 8:
                    q = grid[rr][cc]
                    if q is not None and q.team == t and q.piece_type is not KERNEL:
                        guards += 1
        g = 0.6 * min(guards, 1) + 0.3 * min(max(guards - 1, 0), 1) + 0.1 * min(max(guards - 2, 0), 2)
        rg = ring_of(r, c)
        k = g - 0.15 * rg
        if lr is not None and rg >= lr:
            k -= 4.0 if rem == 0 else 2.0 if rem == 1 else 0.6 if rem == 2 else 0.15
        s += sg * k
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
        acts = legal_actions_for_team(board, team)
        if not acts:
            return static_eval(board, team, rnd)
        acts, _ = ordered(board, team, acts)
        best = -1e9
        for a in acts:
            v = self.child_value(board, team, al, rnd, a, depth, alpha, beta, False)
            if v > best:
                best = v
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        return best

    def root(self, board, team, al, rnd, depth=None, margin=3.0):
        """Scores for every legal action (moves worse than best-margin get an upper bound)."""
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
