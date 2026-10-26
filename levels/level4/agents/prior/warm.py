"""Warmup-only prior sharpening wrapper (ONNX graph surgery, value untouched).

policy_out = policy * (1 + a*r1 + b*r2)
  r1 = |plane14 - 0.125| < tol AND sum(plane12) == 0   (round 1)
  r2 = |plane14 - 0.25 | < tol AND sum(plane12) == 0   (round 2)
Ops: Flatten, MatMul, Sub, Abs, Less, Cast, Mul, Add.

    uv run python levels/level4/agents/prior/warm.py <src.onnx> <dst.onnx> <a> <b> [--novalidate]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def build_warm(src, dst, a: float, b: float, tol: float = 0.01):
    m = onnx.load(str(src))
    g = m.graph
    obs = g.input[0].name
    pol = g.output[0].name
    assert pol == "policy"
    prod = [n for n in g.node if pol in n.output]
    assert len(prod) == 1
    n0 = prod[0]
    n0.output[list(n0.output).index(pol)] = "warm_policy_raw"
    for n in g.node:  # any other consumer of the raw policy (none expected)
        for i, x in enumerate(n.input):
            if x == pol:
                n.input[i] = "warm_policy_raw"
    W = np.zeros((16, 64, 3), np.float32)
    W[14, :, 0] = 1.0 / 64; W[14, :, 1] = 1.0 / 64; W[12, :, 2] = 1.0
    inits = {
        "warm_W": W.reshape(1024, 3),
        "warm_C": np.array([0.125, 0.25, 0.0], np.float32),
        "warm_T": np.array([tol, tol, 0.5], np.float32),
        "warm_Wab": np.array([[a], [b], [0.0]], np.float32),
        "warm_W001": np.array([[0.0], [0.0], [1.0]], np.float32),
        "warm_one": np.array([1.0], np.float32),
    }
    for k, v in inits.items():
        g.initializer.append(numpy_helper.from_array(v, k))
    nodes = [
        helper.make_node("Flatten", [obs], ["warm_xf"], axis=1),
        helper.make_node("MatMul", ["warm_xf", "warm_W"], ["warm_s"]),
        helper.make_node("Sub", ["warm_s", "warm_C"], ["warm_d"]),
        helper.make_node("Abs", ["warm_d"], ["warm_ad"]),
        helper.make_node("Less", ["warm_ad", "warm_T"], ["warm_lt"]),
        helper.make_node("Cast", ["warm_lt"], ["warm_ind"], to=TensorProto.FLOAT),
        helper.make_node("MatMul", ["warm_ind", "warm_Wab"], ["warm_ab"]),
        helper.make_node("MatMul", ["warm_ind", "warm_W001"], ["warm_nl"]),
        helper.make_node("Mul", ["warm_ab", "warm_nl"], ["warm_g"]),
        helper.make_node("Add", ["warm_g", "warm_one"], ["warm_f"]),
        helper.make_node("Mul", ["warm_policy_raw", "warm_f"], [pol]),
    ]
    g.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))
    return len(g.node), len(g.initializer)


if __name__ == "__main__":
    src, dst, a, b = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    nn_, ni = build_warm(src, dst, a, b)
    print(f"{dst}: {nn_} nodes, {ni} initializers")
    if "--novalidate" not in sys.argv:
        from c1 import validate
        for k in range(4):
            ok, msg = validate(dst)
            print(msg[-600:])
            if ok:
                break
