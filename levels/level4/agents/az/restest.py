if __name__ == "__main__":
    import sys, time
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np, torch
    import reslib
    net = reslib.ResModel(32, 3)
    p = reslib.HERE / "exp" / "res_test.onnx"
    reslib.export(net, p)
    print("parity", reslib.parity(p))
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    for q in (p, reslib.HAND):
        s = ort.InferenceSession(str(q), so, providers=["CPUExecutionProvider"])
        x = (np.random.rand(8, 16, 8, 8) > 0.9).astype(np.float32)
        t = time.perf_counter()
        for _ in range(200): s.run(None, {"observation": x})
        print(q.name, "ms/call b8", (time.perf_counter() - t) / 200 * 1000, q.stat().st_size)
    print(reslib.validate(p)[0])
