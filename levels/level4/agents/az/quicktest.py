import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch, numpy as np
from az import Net, export, MODELS
from c1 import validate
if __name__ == "__main__":
    for c, b in [(32, 4), (48, 4)]:
        net = Net(c, b)
        p = MODELS / f"az_test_{c}x{b}.onnx"
        t = time.time(); export(net, p, check=False)
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        s = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
        x = np.random.rand(8, 16, 8, 8).astype(np.float32)
        pol, v = s.run(None, {"observation": x})
        tp, tv = net.eval()(torch.from_numpy(x))
        print(c, b, pol.shape, v.shape, np.abs(pol - tp.detach().numpy()).max(), sum(q.numel() for q in net.parameters()), p.stat().st_size)
        t = time.perf_counter()
        for _ in range(300): s.run(None, {"observation": x})
        print("ms per batch-8 call", (time.perf_counter() - t) / 300 * 1000)
    from sp import selfplay_game
    t = time.process_time()
    r = selfplay_game((str(MODELS / "az_test_32x4.onnx"), 100, 1, 12, 0.3))
    print("game n", r["n"], "cpu", r["cpu"], "L", max(len(a) for a in r["idx"]))
    r = selfplay_game((str(MODELS / "seat_l3_material.onnx"), 100, 1, 12, 0.3))
    print("seat game n", r["n"], "cpu", r["cpu"])
    ok, msg = validate(MODELS / "az_test_32x4.onnx"); print(ok, msg[-600:])
