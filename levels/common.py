"""COLONY: ONE game engine shared by the levels.

Board: dict {(x, y): piece}, x=0..7 is A..H, y=0..7 is rank 1..8.
Pieces: "K"/"P" = CYAN, "k"/"p" = MAGENTA.
Rules implemented: L1 moves/captures, L2 bodyguard, firewall, kernel endgame.
"""

from functools import lru_cache

FILES = "ABCDEFGH"
KING_STEPS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
ORTHO = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
FIREWALL = {8: 0, 16: 1, 24: 2, 26: 3}  # end of round -> ring that disappears
CYAN, MAGENTA = True, False


def name(p):
    return f"{FILES[p[0]]}{p[1] + 1}"


def parse(scene: dict[str, str]):
    return frozenset(((FILES.index(s[0].upper()), int(s[1]) - 1), pc) for s, pc in scene.items())


def ring(p):
    """0 = outer edge, 3 = central 2x2."""
    return min(p[0], p[1], 7 - p[0], 7 - p[1])


def is_cyan(pc):
    return pc.isupper()


def kernel_of(board, team):
    k = "K" if team else "k"
    for p, pc in board.items():
        if pc == k:
            return p
    return None


def alive(p, min_ring):
    return 0 <= p[0] < 8 and 0 <= p[1] < 8 and ring(p) >= min_ring


def attack(board, src, dst, moves_onto):
    """Return list of resulting boards for src attacking enemy piece at dst."""
    target = board[dst]
    if target.upper() == "K":
        guards = [(dst[0] + dx, dst[1] + dy) for dx, dy in KING_STEPS]
        guards = [g for g in guards if g in board and is_cyan(board[g]) == is_cyan(target)]
        if guards:  # bodyguard: guard dies, attacker stays put
            out = []
            for g in guards:
                nb = dict(board)
                del nb[g]
                out.append(nb)
            return out
    nb = dict(board)
    del nb[dst]
    if moves_onto:
        nb[dst] = nb.pop(src)
    return [nb]


def actions(board, team, min_ring):
    """Yield (description, new_board_dict) for every single action of `team`."""
    for p, pc in list(board.items()):
        if is_cyan(pc) != team:
            continue
        x, y = p
        if pc.upper() == "K":
            for dx, dy in KING_STEPS:
                q = (x + dx, y + dy)
                if not alive(q, min_ring):
                    continue
                t = board.get(q)
                if t is None:
                    nb = dict(board)
                    nb[q] = nb.pop(p)
                    yield f"K {name(p)}->{name(q)}", nb
                elif is_cyan(t) != team:
                    for nb in attack(board, p, q, True):
                        yield f"K {name(p)}x{name(q)}", nb
        else:
            for dx, dy in ORTHO:
                for d in (1, 2):
                    q = (x + dx * d, y + dy * d)
                    if not alive(q, min_ring) or q in board:
                        break
                    nb = dict(board)
                    nb[q] = nb.pop(p)
                    yield f"P {name(p)}->{name(q)}", nb
            for dx, dy in DIAG:
                for d in (1, 2):
                    q = (x + dx * d, y + dy * d)
                    if not alive(q, min_ring):
                        break
                    t = board.get(q)
                    if t:
                        if is_cyan(t) != team:
                            for nb in attack(board, p, q, True):
                                yield f"P {name(p)}x{name(q)}", nb
                        break


def end_of_turn(board, team, rnd):
    """Apply end-of-turn rules. Returns (winner or None, board)."""
    if team == MAGENTA and rnd in FIREWALL:
        r = FIREWALL[rnd]
        board = {p: pc for p, pc in board.items() if ring(p) > r}
        ck, mk = kernel_of(board, CYAN), kernel_of(board, MAGENTA)
        if mk is None:
            return (CYAN if ck else MAGENTA), board
        if ck is None:
            return MAGENTA, board
    if len(board) == 2 and kernel_of(board, CYAN) and kernel_of(board, MAGENTA):
        ck, mk = kernel_of(board, CYAN), kernel_of(board, MAGENTA)
        if max(abs(ck[0] - mk[0]), abs(ck[1] - mk[1])) <= 2:
            return (not team), board
        return (CYAN if ring(ck) > ring(mk) else MAGENTA), board
    return None, board


def min_ring_for(rnd):
    """Ring index below which tiles are already gone during round `rnd`."""
    return sum(1 for r in FIREWALL if r < rnd)


def turns(state, team, rnd):
    """All 2-action turns. Yields (desc, winner_or_None, end_state)."""
    board = dict(state)
    mr = min_ring_for(rnd)
    enemy = not team
    for d1, b1 in actions(board, team, mr):
        if kernel_of(b1, enemy) is None:
            yield d1, team, frozenset(b1.items())
            continue
        for d2, b2 in actions(b1, team, mr):
            if kernel_of(b2, enemy) is None:
                yield f"{d1}, {d2}", team, frozenset(b2.items())
                continue
            w, b3 = end_of_turn(b2, team, rnd)
            yield f"{d1}, {d2}", w, frozenset(b3.items())


@lru_cache(maxsize=None)
def _cyan_turns(state, rnd):
    seen, out = set(), []
    for d, w, s in turns(state, CYAN, rnd):
        if (w, s) not in seen:
            seen.add((w, s))
            out.append((d, w, s))
    # try wins first, then moves leaving MAGENTA few replies
    out.sort(key=lambda t: (t[1] is not CYAN, ))
    return out


@lru_cache(maxsize=None)
def _magenta_turns(state, rnd):
    seen, out = set(), []
    for d, w, s in turns(state, MAGENTA, rnd):
        if (w, s) not in seen:
            seen.add((w, s))
            out.append((d, w, s))
    out.sort(key=lambda t: (t[1] is not MAGENTA, ))
    return out


@lru_cache(maxsize=None)
def cyan_wins(state, rnd, depth):
    """CYAN to move in round `rnd`; can CYAN force a win within `depth` CYAN turns?"""
    return best_cyan_turn(state, rnd, depth) is not None


@lru_cache(maxsize=None)
def best_cyan_turn(state, rnd, depth):
    if depth == 0:
        return None
    for d, w, s in _cyan_turns(state, rnd):
        if w is CYAN:
            return d, s
        if w is MAGENTA:
            continue
        if depth == 1:
            continue
        if all(mw is CYAN or (mw is None and cyan_wins(ms, rnd + 1, depth - 1))
               for _, mw, ms in _magenta_turns(s, rnd)):
            return d, s
    return None


def solve_scene(scene: dict[str, str], rnd: int = 1, max_depth: int = 6):
    state = parse(scene)
    for depth in range(1, max_depth + 1):
        res = best_cyan_turn(state, rnd, depth)
        if res:
            return depth, res[0]
        print(f"  no forced win in {depth} CYAN turn(s)")
    return None, None
