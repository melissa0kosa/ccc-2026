import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import torch
from train import make_net
from c1 import export_onnx
import onnx
tmp = HERE / "tmp_models"; tmp.mkdir(exist_ok=True)
for arch, c, b in [("hbs", 32, 3), ("hbs", 24, 3), ("hbs", 32, 2), ("hb", 24, 3), ("hbs", 16, 4)]:
    net = make_net(arch, c, b).eval()
    p = tmp / f"{arch}_{c}x{b}.onnx"
    export_onnx(net, p, check=False)
    m = onnx.load(str(p))
    d = m.graph.output[0].type.tensor_type.shape.dim[1]; d.ClearField("dim_param"); d.dim_value = 4096
    onnx.save(m, str(p))
    print(p)
