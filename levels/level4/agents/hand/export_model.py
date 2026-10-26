"""Export a HandNet with given weights to ONNX. Usage: export_model.py v1  (uses V1_WEIGHTS)
or export_model.py vN weights.json"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
import hand
from c1 import export_onnx, MODELS, validate


def export(name, weights, bias=0.0, pol=None):
    import fast
    net = fast.FastNet(weights, bias, pol).eval()
    p = MODELS / f"hand_{name}.onnx"
    export_onnx(net, p, check=False)
    import onnx
    m = onnx.load(str(p))
    d = m.graph.output[0].type.tensor_type.shape.dim[1]
    d.ClearField("dim_param"); d.dim_value = 4096
    onnx.save(m, str(p))
    ok, msg = validate(p)
    if not ok:
        raise RuntimeError(msg)
    import onnxruntime as ort
    s = ort.InferenceSession(str(p))
    x = np.zeros((8, 16, 8, 8), np.float32)
    t = time.perf_counter()
    for _ in range(200):
        s.run(None, {"observation": x})
    print(f"nodes {len(m.graph.node)} initializers {len(m.graph.initializer)}")
    print(f"exported {p} ({p.stat().st_size} B), {1e3*(time.perf_counter()-t)/200:.3f} ms/call b8")
    return p


if __name__ == "__main__":
    name = sys.argv[1]
    if len(sys.argv) > 2:
        d = json.loads(Path(sys.argv[2]).read_text())
        export(name, d["weights"], d.get("bias", 0.0), d.get("pol"))
    else:
        export(name, hand.V1_WEIGHTS, 0.0)
