"""Export via c1.export_onnx, then pin the policy output's static dim to 4096 and validate."""
import sys
from pathlib import Path
L4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(L4))
from c1 import export_onnx, validate  # noqa: E402


def export(model, path):
    import onnx
    path = Path(path)
    export_onnx(model, path, check=False)
    m = onnx.load(str(path))
    for o in m.graph.output:
        if o.name == "policy":
            d = o.type.tensor_type.shape.dim[1]
            d.ClearField("dim_param")
            d.dim_value = 4096
    onnx.checker.check_model(m)
    onnx.save(m, str(path))
    for attempt in range(3):
        ok, msg = validate(path)
        if ok:
            return path
        if "too slow" not in msg:
            break
    raise RuntimeError(f"validator rejected {path}:\n{msg}")


if __name__ == "__main__":
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train import Net
    net = Net(int(sys.argv[3]), int(sys.argv[4]))
    net.load_state_dict(torch.load(sys.argv[1]))
    print(export(net.eval(), sys.argv[2]))
