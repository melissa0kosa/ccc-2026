"""Export a FastNet4 model. usage: export4.py name weights.json [pol overrides k:v,...] [w overrides k:v,...]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np


def export(name, weights, bias, pol):
    import fast5 as fast4, onnx
    from c1 import export_onnx, MODELS, validate
    net = fast4.FastNet5(weights, bias, pol).eval()
    p = MODELS / f"hand_{name}.onnx"
    export_onnx(net, p, check=False)
    m = onnx.load(str(p))
    d = m.graph.output[0].type.tensor_type.shape.dim[1]
    if d.dim_value != 4096:
        d.ClearField("dim_param"); d.dim_value = 4096; onnx.save(m, str(p))
    ok, msg = validate(p)
    if not ok:
        raise RuntimeError(msg)
    dry = [ln for ln in msg.splitlines() if "dry run" in ln]
    print(f"nodes {len(m.graph.node)} initializers {len(m.graph.initializer)} | {dry[0].strip() if dry else ''}")
    out = {"weights": weights, "bias": bias, "pol": dict(net.pv)}
    (Path(__file__).resolve().parent / "data" / f"w_{name}.json").write_text(json.dumps(out, indent=1))
    print("exported", p)


if __name__ == "__main__":
    name, wj = sys.argv[1], json.loads(Path(sys.argv[2]).read_text())
    pol = dict(wj.get("pol") or {})
    w = dict(wj["weights"])
    if len(sys.argv) > 3 and sys.argv[3]:
        for it in sys.argv[3].split(","):
            k, v = it.split(":"); pol[k] = float(v)
    if len(sys.argv) > 4 and sys.argv[4]:
        for it in sys.argv[4].split(","):
            k, v = it.split(":"); w[k] = float(v)
    export(name, w, wj.get("bias", 0.0), pol)
