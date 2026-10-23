"""What-if: at critical decisions of a live game, what would other models have played?

uv run python levels/level4/whatif.py
Replays each listed (match_id, k) position (k = actions already played), then for each model
prints value, prior top move, and the 100-sim referee choices over 5 seeds; plus an 800-sim reference.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_matches import parse  # noqa: E402
from c1 import MODELS, ColonyOneEnv, SearchAgent, encoding, evaluator  # noqa: E402

MATCHES = Path(__file__).parent / "data" / "matches" / "matches_x3eg_a.json"
CASES = [(22984, 22, "our Daemon F4->C4->shoot C5 (then lost to K walk)"),
         (22984, 26, "our Trace shot strips their last guard, Trace left on E6"),
         (23081, 24, "our Ice C2->E4->E5 into Daemon slide")]
MODEL_NAMES = ["az_res4f_x3_eg", "az_res4f_v6"]


def position(m, k):
    moves, _ = parse(m["protocol"])
    env = ColonyOneEnv()
    for team, _p, a, b, _kind in moves[:k]:
        env.step([x for x in env.legal_actions() if env.format_action(x).startswith(f"{a} -> {b}")][0])
    played = moves[k]
    return env, f"{played[2]} -> {played[3]}"


def main():
    games = {m["match_id"]: m for m in json.loads(MATCHES.read_text())}
    for mid, k, note in CASES:
        env, played = position(games[mid], k)
        print(f"\n== {mid} action {k + 1} ({env.current_team}, {env.actions_left} left): {note}; live played {played}")
        acts = {encoding.action_index(x, env.current_team): env.format_action(x) for x in env.legal_actions()}
        for name in MODEL_NAMES:
            ev = evaluator(str((MODELS / f"{name}.onnx").resolve()))
            lg, v = ev(env.observation()[None])
            lg = lg[0].copy()
            lg[~env.legal_mask()] = -1e9
            p = np.exp(lg - lg.max())
            p /= p.sum()
            top = int(np.argmax(p))
            picks = [env.format_action(SearchAgent(ev, 100, seed=s).select(
                env.board, env.current_team, env.actions_left, env.round_number)) for s in range(5)]
            print(f"   {name:16s} value {v[0]:+.2f} prior-top {acts[top]} ({p[top]:.2f}) 100-sim picks {picks}")
        ref = SearchAgent(evaluator(str((MODELS / "az_res4f_v6.onnx").resolve())), 800, seed=0)
        best = ref.select(env.board, env.current_team, env.actions_left, env.round_number)
        print(f"   reference v6@800 picks {env.format_action(best)}")


if __name__ == "__main__":
    main()
