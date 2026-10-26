import sys, onnxruntime as ort, numpy as np, time, onnx, json, os
from collections import Counter, defaultdict
sys.path.insert(0, 'levels/level4/agents/hand'); import fit
X, Z, IDX, CNT = fit.load(['levels/level4/agents/hand/data/sp_v1.pkl'])
for p in sys.argv[1:]:
    m = onnx.load(p)
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    so.enable_profiling = '--prof' in os.environ.get('PROF', '')
    s = ort.InferenceSession(p, so, providers=['CPUExecutionProvider'])
    res = []
    for b in (1, 8, 16):
        xb = X[:b]
        t = time.perf_counter()
        for i in range(300): s.run(None, {'observation': X[(i*b) % 2000:(i*b) % 2000 + b]})
        res.append(f"b{b} {(time.perf_counter()-t)/300*1e3:.3f}ms")
    print(os.path.basename(p), len(m.graph.node), "nodes", *res)
