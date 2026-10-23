"""Sweep stopgap value/prior scaling vs the live stopgap_v1 (60 games each, 4 workers).

uv run python levels/level4/sweep_stopgap.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1 import MODELS, arena, export_onnx  # noqa: E402
from stopgap import Stopgap  # noqa: E402

VARIANTS = {
    "v05": dict(v_scale=0.5),
    "v2": dict(v_scale=2.0),
    "p0": dict(p_scale=0.0),
    "p2": dict(p_scale=2.0),
    "p4": dict(p_scale=4.0),
}

if __name__ == "__main__":
    base = MODELS / "stopgap_v1.onnx"
    out = HERE = Path(__file__).parent / "sweep_results.jsonl"
    for name, kw in VARIANTS.items():
        p = export_onnx(Stopgap(**kw), MODELS / "sweep" / f"stopgap_{name}.onnx", check=False)
        r = arena(p, base, games=60, workers=4)
        with out.open("a") as fh:
            fh.write(json.dumps({"variant": name, **kw, **{k: r[k] for k in
                                 ("wins", "games", "rate", "ci", "cyan_rate", "magenta_rate",
                                  "cpu_per_game")}}) + "\n")
