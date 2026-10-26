"""Scratch: replay a live game to action k and query stopgap_v1 / hand_v2 there.
usage: uv run python levels/level4/reviews/sg_pos.py MATCH_ID K1 [K2 ...] [--mcts N]
K = number of protocol actions already applied (position BEFORE protocol action K+1)."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from c1 import evaluator, mcts, MODELS, ColonyOneEnv
from colony_one.encoding import action_index
from analyze_matches import parse

SG = evaluator(str(MODELS / "stopgap_v1.onnx"))
HV = evaluator(str(MODELS / "hand_v2.onnx"))

def env_at(moves, k):
    env = ColonyOneEnv()
    for team, _p, a, b, _k in moves[:k]:
        cands = [x for x in env.legal_actions() if env.format_action(x).replace(" ", "") == f"{a}->{b}"]
        assert len(cands) == 1, (a, b)
        env.step(cands[0])
    return env

def show(env, ev, name, nxt=None, top=6):
    obs = env.observation()
    logits, v = ev(obs[None])
    acts = env.legal_actions()
    idx = np.array([action_index(a, env.current_team) for a in acts])
    lg = logits[0][idx]
    p = np.exp(lg - lg.max()); p /= p.sum()
    order = np.argsort(-p)
    s = f"  {name:8s} v={v[0]:+.3f} top: " + ", ".join(f"{env.format_action(acts[i]).replace(' ', '')}:{p[i]:.2f}" for i in order[:top])
    if nxt:
        j = [i for i, a in enumerate(acts) if env.format_action(a).replace(" ", "") == nxt]
        if j:
            r = int(np.where(order == j[0])[0][0]) + 1
            s += f" | played {nxt} p={p[j[0]]:.2f} rank {r}/{len(acts)}"
    print(s)

def visits(env, ev, sims=100, seeds=4):
    out = {}
    for s in range(seeds):
        a = mcts.search_action(env.board, env.current_team, env.actions_left, env.round_number, ev,
                               mcts.MCTSConfig(simulations=sims), rng=np.random.default_rng(s))
        k = env.format_action(a).replace(" ", "")
        out[k] = out.get(k, 0) + 1
    return out

if __name__ == "__main__":
    mid = int(sys.argv[1]); ks = [int(x) for x in sys.argv[2:] if not x.startswith("--")]
    do_mcts = "--mcts" in sys.argv
    files = ["matches_1405_stopgap_v1.json", "tournament-matches.json"]
    m = next(x for f in files for x in json.loads((HERE / "data/matches" / f).read_text()) if x["match_id"] == mid)
    moves, _ = parse(m["protocol"])
    for k in ks:
        env = env_at(moves, k)
        nxt = f"{moves[k][2]}->{moves[k][3]}" if k < len(moves) else None
        print(f"--- match {mid} after {k} actions: {env.current_team} to move, actions_left={env.actions_left}, round={env.round_number}, next played: {moves[k][:4] if nxt else None}")
        if "--board" in sys.argv:
            print(env.render())
        show(env, SG, "stopgap", nxt)
        show(env, HV, "hand_v2", nxt)
        if do_mcts:
            print("   mcts100 choice (4 seeds) stopgap:", visits(env, SG), " hand_v2:", visits(env, HV))
