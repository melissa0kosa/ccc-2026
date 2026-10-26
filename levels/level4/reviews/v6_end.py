"""Scratch: print protocol tail + final boards for given matches."""
import sys
sys.path.insert(0, "levels/level4"); sys.path.insert(0, "levels/level4/reviews")
from v6_common import load
from analyze_matches import parse
from sg_pos import env_at
ids = [int(x) for x in sys.argv[1:]]
for m in load():
    if m["match_id"] not in ids: continue
    moves, ev = parse(m["protocol"])
    L = [l for l in m["protocol"].splitlines() if l.strip()]
    print("=====", m["match_id"], m["you"], "n", len(moves)); print("\n".join(L[-14:]))
    for k in (len(moves) - 6, len(moves) - 2, len(moves) - 1):
        env = env_at(moves, k); print(f"-- before a{k+1} {env.current_team} al{env.actions_left} r{env.round_number} next {moves[k][:4]}"); print(env.render())
