"""Inference-time derivations of a champion ONNX (graph surgery, no training).

build(src, dst, syms=(0,), pol_scale=1.0, val_scale=1.0, warm=None)
  * TTA: stack the observation with D4 transforms along batch, run the graph once,
    gather each policy back to canonical action indices, average logits; value =
    tanh(val_scale * mean(pre-tanh value)).
  * pol_scale: global logit multiplier (all rounds).
  * warm=(a, b): round-1/2 logit factor (1+a) / (1+b) via agents/prior/warm.py.

D4 convention (azlib / symcheck): sym s maps square sq -> PERM[s, sq];
transformed obs out[:, j] = obs[:, INV[s, j]]; action a -> IDX[s, a].
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

HERE = Path(__file__).resolve().parent
L4 = HERE.parents[1]
sys.path.insert(0, str(L4))
sys.path.insert(0, str(L4 / "agents" / "prior"))


def perms():
    P = np.zeros((8, 64), np.int64)
    for s in range(8):
        for sq in range(64):
            r, c = divmod(sq, 8)
            if s & 4:
                r, c = c, r
            if s & 1:
                c = 7 - c
            if s & 2:
                r = 7 - r
            P[s, sq] = r * 8 + c
    return P


PERM = perms()
INV = np.argsort(PERM, axis=1)
IDX = (PERM[:, :, None] * 64 + PERM[:, None, :]).reshape(8, 4096)


def tf_obs(obs, s):  # numpy reference, obs (..., 16, 8, 8)
    flat = obs.reshape(*obs.shape[:-2], 64)
    return flat[..., INV[s]].reshape(obs.shape)


def build(src, dst, syms=(0,), pol_scale=1.0, val_scale=1.0, warm=None, res_scale=None):
    m = onnx.load(str(src))
    g = m.graph
    if res_scale is not None:  # az_res4f_v2: dp = /Reshape_4_output_0, dv = /Reshape_5_output_0
        ap, av = (res_scale, res_scale) if np.isscalar(res_scale) else res_scale
        for t, a, nm in (("/Reshape_4_output_0", ap, "d_rsp"), ("/Reshape_5_output_0", av, "d_rsv")):
            if a == 1.0:
                continue
            cons = [n for n in g.node if t in n.input]
            assert len(cons) == 1 and cons[0].op_type == "Add", cons
            i = list(cons[0].input).index(t)
            cons[0].input[i] = nm + "_o"
            g.initializer.append(numpy_helper.from_array(np.array([a], np.float32), nm))
            idx = list(g.node).index(cons[0])
            g.node.insert(idx, helper.make_node("Mul", [t, nm], [nm + "_o"]))
    obs = g.input[0].name
    pol, val = g.output[0].name, g.output[1].name
    k = len(syms)
    assert syms[0] == 0
    tanh = [n for n in g.node if val in n.output]
    assert len(tanh) == 1 and tanh[0].op_type == "Tanh"
    pre = tanh[0].input[0]
    g.node.remove(tanh[0])
    ppro = [n for n in g.node if pol in n.output]
    assert len(ppro) == 1
    ppro[0].output[list(ppro[0].output).index(pol)] = "d_pol_raw"
    inits, nodes = {}, []
    # ---- input stacking
    if k > 1:
        inner = "d_x"
        for n in g.node:
            for i, x in enumerate(n.input):
                if x == obs:
                    n.input[i] = inner
        inits["d_shp_f"] = np.array([-1, 16, 64], np.int64)
        inits["d_shp_b"] = np.array([-1, 16, 8, 8], np.int64)
        nodes.append(helper.make_node("Reshape", [obs, "d_shp_f"], ["d_xf"]))
        cat = [obs]
        for s in syms[1:]:
            inits[f"d_inv{s}"] = INV[s].astype(np.int64)
            nodes.append(helper.make_node("Gather", ["d_xf", f"d_inv{s}"], [f"d_xg{s}"], axis=2))
            nodes.append(helper.make_node("Reshape", [f"d_xg{s}", "d_shp_b"], [f"d_xt{s}"]))
            cat.append(f"d_xt{s}")
        nodes.append(helper.make_node("Concat", cat, [inner], axis=0))
        # ---- policy unstack
        inits["d_shp_p"] = np.array([k, -1, 4096], np.int64)
        nodes.append(helper.make_node("Reshape", ["d_pol_raw", "d_shp_p"], ["d_pk"]))
        parts = []
        for i, s in enumerate(syms):
            inits[f"d_i{i}"] = np.array(i, np.int64)
            nodes.append(helper.make_node("Gather", ["d_pk", f"d_i{i}"], [f"d_p{i}"], axis=0))
            if s == 0:
                parts.append(f"d_p{i}")
            else:
                inits[f"d_idx{s}"] = IDX[s].astype(np.int64)
                nodes.append(helper.make_node("Gather", [f"d_p{i}", f"d_idx{s}"], [f"d_pm{i}"], axis=1))
                parts.append(f"d_pm{i}")
        nodes.append(helper.make_node("Sum", parts, ["d_psum"]))
        pol_in, pol_mul = "d_psum", pol_scale / k
        # ---- value unstack
        inits["d_shp_v"] = np.array([k, -1], np.int64)
        nodes.append(helper.make_node("Reshape", [pre, "d_shp_v"], ["d_vk"]))
        nodes.append(helper.make_node("ReduceMean", ["d_vk"], ["d_vm"], axes=[0], keepdims=0))
        v_in = "d_vm"
    else:
        pol_in, pol_mul, v_in = "d_pol_raw", pol_scale, pre
    if pol_mul != 1.0:
        inits["d_pmul"] = np.array([pol_mul], np.float32)
        nodes.append(helper.make_node("Mul", [pol_in, "d_pmul"], [pol]))
    else:
        nodes.append(helper.make_node("Identity", [pol_in], [pol]))
    if val_scale != 1.0:
        inits["d_vmul"] = np.array([val_scale], np.float32)
        nodes.append(helper.make_node("Mul", [v_in, "d_vmul"], ["d_vs"]))
        v_in = "d_vs"
    nodes.append(helper.make_node("Tanh", [v_in], [val]))
    for kk, v in inits.items():
        g.initializer.append(numpy_helper.from_array(v, kk))
    # Prepend input-side nodes (topological order): stacking nodes first.
    pre_nodes = [n for n in nodes if n.output[0].startswith("d_x") or n.output[0] == "d_x"]
    post_nodes = [n for n in nodes if n not in pre_nodes]
    old = list(g.node)
    del g.node[:]
    g.node.extend(pre_nodes + old + post_nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))
    if warm is not None:
        from warm import build_warm
        build_warm(dst, dst, warm[0], warm[1])
    from c1 import fix_constants
    fix_constants(Path(dst))
    m = onnx.load(str(dst))
    return len(m.graph.node), len(m.graph.initializer)


def check(src, dst, n=64, syms=(0,), seed=0):
    """Reference check vs numpy averaging of src, and equivariance under the TTA group."""
    import onnxruntime as ort
    import pickle
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    A = ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])
    B = ort.InferenceSession(str(dst), so, providers=["CPUExecutionProvider"])
    f = sorted((L4 / "agents" / "az" / "data").glob("games_*.pkl"))[0]
    games = pickle.load(open(f, "rb"))
    x = np.concatenate([gm["obs"] for gm in games[:20]]).astype(np.float32) / 8
    rng = np.random.default_rng(seed)
    x = x[rng.choice(len(x), n, replace=False)]
    pb, vb = B.run(None, {"observation": x})
    ps, vs = [], []
    for s in syms:
        p, v = A.run(None, {"observation": tf_obs(x, s)})
        ps.append(p[:, IDX[s]]); vs.append(np.arctanh(np.clip(v, -0.999999, 0.999999)))
    pr = np.mean(ps, 0); vr = np.tanh(np.mean(vs, 0))
    out = dict(dpol=float(np.abs(pb - pr).max()), dval=float(np.abs(vb - vr).max()))
    for s in syms[1:]:
        p2, v2 = B.run(None, {"observation": tf_obs(x, s)})
        out[f"eq{s}"] = (float(np.abs(p2[:, IDX[s]] - pb).max()), float(np.abs(v2 - vb).max()))
    return out


if __name__ == "__main__":
    import json
    src = sys.argv[1]; dst = sys.argv[2]
    kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    if "syms" in kw:
        kw["syms"] = tuple(kw["syms"])
    print(dst, build(src, dst, **kw))
