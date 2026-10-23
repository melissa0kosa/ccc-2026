"""Turn our downloaded live tournament games into training records (sp_v4f.pkl format).

Each record: (planes 0-12 uint8 (13,8,8), planes 13-15 float32 (3,), z, indices int16, counts int16, rootQ)
- policy target = one-hot of the move actually played; z = rootQ = game result for the side to move.
- Opponent moves (real other teams) -> live_opp.pkl, our moves -> live_ours.pkl.

Run with PyCharm ▶ or:  uv run python levels/level4/live_to_records.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_matches import parse  # noqa: E402
from c1 import ColonyOneEnv, encoding  # noqa: E402

HERE = Path(__file__).parent
MATCHES = HERE / "data" / "matches"
OUT = HERE / "agents" / "live"


def find_action(env, a, b):
    want = f"{a} -> {b}"
    cands = [x for x in env.legal_actions() if env.format_action(x).startswith(want)]
    if len(cands) != 1:
        raise ValueError(f"cannot match {want}: {[env.format_action(x) for x in cands]}")
    return cands[0]


def load_games():
    games = {}
    for f in sorted(MATCHES.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, list):
            continue
        for m in data:
            if isinstance(m, dict) and m.get("protocol") and m.get("match_id") not in games:
                games[m["match_id"]] = m
    return games


def main():
    games = load_games()
    opp, ours, bad = [], [], 0
    for mid, m in games.items():
        moves, _ = parse(m["protocol"])
        winner = "CYAN" if "CYAN WINS" in m["protocol"] else "MAGENTA"
        env = ColonyOneEnv()
        recs = []
        try:
            for team, _piece, a, b, _kind in moves:
                assert env.current_team == team
                act = find_action(env, a, b)
                obs = env.observation()
                z = 1.0 if team == winner else -1.0
                idx = encoding.action_index(act, team)
                rec = (obs[:13].astype(np.uint8), obs[13:16, 0, 0].astype(np.float32), z,
                       np.array([idx], np.int16), np.array([1], np.int16), z)
                recs.append((team != m["you"], rec))
                env.step(act)
        except Exception:  # noqa: BLE001 - skip games we can't replay exactly
            bad += 1
            continue
        for is_opp, rec in recs:
            (opp if is_opp else ours).append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(opp, open(OUT / "live_opp.pkl", "wb"))
    pickle.dump(ours, open(OUT / "live_ours.pkl", "wb"))
    zo = np.mean([r[2] for r in opp]) if opp else 0
    print(f"games {len(games)} (replay failures {bad}) -> opponent records {len(opp)} "
          f"(mean z {zo:+.2f}), our records {len(ours)}")
    print("wrote", OUT / "live_opp.pkl", "and", OUT / "live_ours.pkl")


if __name__ == "__main__":
    main()
