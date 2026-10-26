import sys, time
import numpy as np
mod = __import__(sys.argv[1])
env = mod.ColonyOneEnv()
rng = np.random.default_rng(1)
S = mod.Searcher()
tt = []; out = []
for i in range(int(sys.argv[2]) if len(sys.argv) > 2 else 40):
    if env.terminated: break
    t = time.perf_counter()
    acts, sc, best = S.root(env.board, env.current_team, env.actions_left, env.round_number)
    tt.append(time.perf_counter() - t); out.append(round(best, 3))
    env.step(acts[int(rng.integers(len(acts)))])
print(sys.argv[1], "total", round(sum(tt), 2), out)
