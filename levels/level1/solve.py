"""Level 1: find CYAN 2-action plays that leave the CYAN Kernel safe.

Edit SCENE below (uppercase = CYAN, lowercase = MAGENTA), then run with PyCharm ▶
or:  uv run python levels/level1/solve.py
"""

from itertools import product

# square -> piece. K/P = CYAN, k/p = MAGENTA
SCENE = {"G1": "K", "E2": "p", "E4": "k"}
SCENE = {sq.upper(): pc for sq, pc in SCENE.items() if pc}

FILES = "ABCDEFGH"
KING_STEPS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
ORTHO = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def sq(x, y):
    return f"{FILES[x]}{y + 1}"


def xy(s):
    return FILES.index(s[0]), int(s[1]) - 1


def inside(x, y):
    return 0 <= x < 8 and 0 <= y < 8


def is_cyan(pc):
    return pc.isupper()


def actions(board, cyan):
    """Yield (description, new_board) for every single action of one team."""
    for s, pc in board.items():
        if is_cyan(pc) != cyan:
            continue
        x, y = xy(s)
        if pc.upper() == "K":
            for dx, dy in KING_STEPS:
                nx, ny = x + dx, y + dy
                if not inside(nx, ny):
                    continue
                t = board.get(sq(nx, ny))
                if t and is_cyan(t) == cyan:
                    continue
                yield f"K {s}->{sq(nx, ny)}" + (" x" + t if t else ""), move(board, s, sq(nx, ny))
        else:  # Ping
            for dx, dy in ORTHO:
                for d in (1, 2):
                    nx, ny = x + dx * d, y + dy * d
                    if not inside(nx, ny) or sq(nx, ny) in board:
                        break
                    yield f"P {s}->{sq(nx, ny)}", move(board, s, sq(nx, ny))
            for dx, dy in DIAG:
                for d in (1, 2):
                    nx, ny = x + dx * d, y + dy * d
                    if not inside(nx, ny):
                        break
                    t = board.get(sq(nx, ny))
                    if t:
                        if is_cyan(t) != cyan:
                            yield f"P {s}x{sq(nx, ny)}", move(board, s, sq(nx, ny))
                        break


def move(board, a, b):
    nb = dict(board)
    nb[b] = nb.pop(a)
    return nb


def has_kernel(board, cyan):
    return ("K" if cyan else "k") in board.values()


def magenta_can_capture(board, n_actions):
    """True if MAGENTA can capture the CYAN Kernel within n_actions."""
    frontier = [board]
    for _ in range(n_actions):
        nxt = []
        for b in frontier:
            for _, nb in actions(b, cyan=False):
                if not has_kernel(nb, cyan=True):
                    return True
                nxt.append(nb)
        frontier = nxt
    return False


def main():
    print("Scene:", SCENE)
    results = []
    for b1 in [SCENE]:
        for d1, b2 in actions(b1, cyan=True):
            if not has_kernel(b2, cyan=False):
                results.append((d1, "WIN (captures Kernel)", True, True))
                continue
            for d2, b3 in actions(b2, cyan=True):
                won = not has_kernel(b3, cyan=False)
                safe1 = won or not magenta_can_capture(b3, 1)
                safe2 = won or not magenta_can_capture(b3, 2)
                results.append((f"{d1}, {d2}", "WIN" if won else "", safe1, safe2))
    strict = [r for r in results if r[3]]
    print(f"\nSafe vs full MAGENTA turn (2 actions): {len(strict)}")
    for r in strict:
        print("  ", r[0], r[1])
    loose = [r for r in results if r[2] and not r[3]]
    print(f"\nSafe vs 1 action only (NOT vs 2): {len(loose)}")
    for r in loose[:20]:
        print("  ", r[0])


if __name__ == "__main__":
    main()
