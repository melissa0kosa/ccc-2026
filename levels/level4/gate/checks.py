"""Graph caps + official validator (retry 'too slow' up to 3x). usage: checks.py name.onnx [...]"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import onnx  # noqa: E402
from c1 import MODELS, validate  # noqa: E402

out = HERE / "checks.jsonl"
for name in sys.argv[1:]:
    p = MODELS / name
    m = onnx.load(str(p))
    nodes, inits = len(m.graph.node), len(m.graph.initializer)
    tries, ok, msg = 0, False, ""
    for tries in range(1, 4):
        ok, msg = validate(p)
        if ok or "too slow" not in msg.lower():
            break
    tail = [ln for ln in msg.strip().splitlines() if ln.strip()][-3:]
    r = dict(name=name, nodes=nodes, inits=inits, caps_ok=nodes <= 1024 and inits <= 512,
             validator=ok, tries=tries, tail=tail)
    print(json.dumps(r), flush=True)
    with out.open("a") as f:
        f.write(json.dumps(r) + "\n")
