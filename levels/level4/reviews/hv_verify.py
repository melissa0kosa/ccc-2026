"""Scratch: hand_v2 verification: session-create cost, per-call latency (1 thread), value/prior stats
over all live positions, own-side CPU of 100-sim search over the longest live games."""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "reviews"))
from c1 import MODELS, ColonyOneEnv, mcts, make_agent, play_game
from colony_one.onnx_runner import OnnxEvaluator
from colony_one.encoding import action_index
from analyze_matches import parse
from hv_stats import load
from sg_pos import env_at

P = MODELS / "hand_v2.onnx"
c = time.process_time(); w = time.time(); ev = OnnxEvaluator(P)
print(f"session create: {1e3*(time.process_time()-c):.0f} ms cpu, {1e3*(time.time()-w):.0f} ms wall")
ms = load()
obs, legal = [], []
for m in ms:
    moves, _ = parse(m["protocol"])
    for k in range(len(moves)):
        env = env_at(moves, k)
        obs.append(env.observation()); legal.append([action_index(a, env.current_team) for a in env.legal_actions()])
X = np.stack(obs).astype(np.float32)
print("live positions:", len(X))
for b in (1, 8):
    ev(X[:b]); c = time.process_time(); n = 0
    for i in range(0, 400 * b, b):
        j = i % (len(X) - b); ev(X[j:j + b]); n += 1
    print(f"batch {b}: {1e3*(time.process_time()-c)/n:.2f} ms cpu/call")
L, V = [], []
for i in range(0, len(X), 8):
    l, v = ev(X[i:i + 8]); L.append(l); V.append(v)
L = np.concatenate(L); V = np.concatenate(V)
print(f"value: finite={np.isfinite(V).all()} min={V.min():+.5f} max={V.max():+.5f} |v|>0.9: {np.mean(np.abs(V)>0.9):.1%} |v|>0.97: {np.mean(np.abs(V)>0.97):.1%} mean|v|={np.abs(V).mean():.2f}")
print(f"logits finite={np.isfinite(L).all()} range [{L.min():.2f},{L.max():.2f}]")
top = []
for l, lg in zip(L, legal):
    z = l[lg]; p = np.exp(z - z.max()); p /= p.sum(); top.append((p.max(), len(lg), np.sort(p)[::-1][:3].sum()))
top = np.array(top)
q = np.percentile(top[:, 0], [10, 25, 50, 75, 90])
print("prior top-1 percentiles 10/25/50/75/90:", np.round(q, 3), f"top1<0.10: {np.mean(top[:,0]<0.10):.0%}  top1<0.2: {np.mean(top[:,0]<0.2):.0%}  top1>0.5: {np.mean(top[:,0]>0.5):.0%}",
      f"median top-3 mass {np.median(top[:,2]):.2f}  median legal {np.median(top[:,1]):.0f}")
# own-side CPU of the referee search over the longest live games
ms.sort(key=lambda m: -len(parse(m["protocol"])[0]))
for m in ms[:3]:
    moves, _ = parse(m["protocol"]); cpu = []; calls = [0]
    def cnt(x, _e=ev):
        calls[0] += 1; return _e(x)
    for k in range(len(moves)):
        if moves[k][0] != m["you"]:
            continue
        env = env_at(moves, k); c = time.process_time()
        mcts.search_action(env.board, env.current_team, env.actions_left, env.round_number, cnt,
                           mcts.MCTSConfig(simulations=100), rng=np.random.default_rng(k))
        cpu.append(time.process_time() - c)
    print(f"match {m['match_id']}: {len(moves)} actions, own searches {len(cpu)}: cpu total {sum(cpu):.2f}s, mean {np.mean(cpu)*1e3:.0f} ms, max {max(cpu)*1e3:.0f} ms, calls/search {calls[0]/len(cpu):.1f} -> 52 own actions ~ {52*np.mean(cpu):.1f}s (worst-search x52 = {52*max(cpu):.1f}s)")
# self-play full games (both sides) wall/cpu
cpu, n = [], []
for g in range(4):
    a, b = make_agent(str(P), 100, 2 * g), make_agent(str(P), 100, 2 * g + 1)
    c = time.process_time(); _, k = play_game(a, b); cpu.append(time.process_time() - c); n.append(k)
print("self-play both sides cpu/game:", [round(x, 2) for x in cpu], "actions", n)
