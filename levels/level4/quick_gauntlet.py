"""Quick gauntlet (~1-2 min): validator + short arenas vs older models, to catch regressions.

Run with PyCharm ▶ (edit MODEL) or:  uv run python levels/level4/quick_gauntlet.py <model.onnx> [workers]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1 import MODELS, arena, validate  # noqa: E402

MODEL = MODELS / "az_res4f_v6.onnx"
GAUNTLET = [("greedy", 10), (MODELS / "stopgap_v1.onnx", 20), (MODELS / "hand_v4f.onnx", 20),
            (MODELS / "az_res_v20.onnx", 30), (MODELS / "az_res4f_v2.onnx", 40)]


def main():
    model = Path(sys.argv[1]) if len(sys.argv) > 1 else MODEL
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    t0 = time.time()
    ok = False
    for _ in range(3):  # the validator spuriously fails "too slow" under load
        ok, msg = validate(model)
        if ok:
            break
    print(f"== {model.name}: validator {'ACCEPTED' if ok else 'REJECTED: ' + msg[-300:]}")
    rows = []
    for opp, games in GAUNTLET:
        r = arena(model, opp, games=games, workers=workers, verbose=False)
        rows.append(r)
        print(f"   vs {r['b']:24s} {r['wins']:>3}/{r['games']:<3} = {r['rate']:.0%}  "
              f"[{r['ci'][0]:.0%},{r['ci'][1]:.0%}]  cyan {r['cyan_rate']:.0%} magenta {r['magenta_rate']:.0%}  "
              f"{r['cpu_per_game']:.1f}s cpu/game", flush=True)
    worst = min(rows, key=lambda r: r["rate"])
    print(f"   done in {time.time() - t0:.0f}s; weakest matchup: {worst['b']} {worst['rate']:.0%}")


if __name__ == "__main__":
    main()
