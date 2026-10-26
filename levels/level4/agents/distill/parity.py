"""uv run python levels/level4/agents/distill/parity.py NAME arch ch blocks : ONNX vs torch on data positions."""
import glob, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from train import make_net
import onnxruntime as ort
name, arch, ch, bl = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
net = make_net(arch, ch, bl); net.load_state_dict(torch.load(HERE / f"{name}.pt", map_location="cpu")); net.eval()
d = np.load(sorted(glob.glob(str(HERE / "data" / "r*.npz")))[0])
x = d["obs"][:64].astype(np.float32)
with torch.no_grad():
    p, v = net(torch.tensor(x))
s = ort.InferenceSession(str(HERE.parents[1] / "models" / f"{name}.onnx"), providers=["CPUExecutionProvider"])
po, vo = s.run(None, {"observation": x})
print("max |dp|", np.abs(po - p.numpy()).max(), "max |dv|", np.abs(vo - v.numpy()).max(), "v sample", vo[:6].round(3))
