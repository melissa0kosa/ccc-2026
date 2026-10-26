import sys
sys.path.insert(0, "levels/level4"); sys.path.insert(0, "levels/level4/reviews")
from v6_common import load
from analyze_matches import parse
from sg_pos import env_at
mid = int(sys.argv[1]); ks = [int(x) for x in sys.argv[2:]]
m = next(x for x in load() if x["match_id"] == mid)
moves, _ = parse(m["protocol"])
L = [l for l in m["protocol"].splitlines() if l.strip() and l.strip()[0].isdigit()]
for k in ks:
    env = env_at(moves, k); print(f"-- {mid} before a{k+1} {env.current_team} al{env.actions_left} r{env.round_number} next: {L[k].strip()}"); print("\n".join(env.render().splitlines()[:8]))
