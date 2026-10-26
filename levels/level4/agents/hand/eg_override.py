"""Exact, gated bare-Kernel endgame value override wrapped around an existing ONNX model.

Gate (side to move = "own", canonical planes), actions_left == 1, and NOT (plane14 == 1 and Magenta to move):
 A) own = bare Kernel, enemy = Kernel + exactly 1 non-Kernel piece:
      +1 if our Kernel is adjacent to their Kernel and their last piece is NOT adjacent to their Kernel (unguarded capture)
      +1 if our Kernel can capture their last piece and land at Chebyshev >= 3 from their Kernel on a more central
         ring (equal ring if we are Magenta)  -> bare-Kernel rule resolves at our turn end
      else -0.95
      prior veto -5 on suicide captures: last piece onto a non-winning square, or the guarded enemy Kernel
 B) bare Kernel vs bare Kernel: +1 if we can capture their Kernel, or step to a square at Chebyshev >= 3 on a more
      central ring (equal ring if Magenta); else -1
Outside the gate the output is bit-identical to the base model (Where on value and policy).

usage: uv run python levels/level4/agents/hand/eg_override.py base.onnx out.onnx
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

VETO = -5.0
LOSE_A = -0.95


class EGOverride(nn.Module):
    def __init__(self):
        super().__init__()
        ring = np.ones((1, 1, 3, 3), np.float32); ring[0, 0, 1, 1] = 0
        self.register_buffer("w_adj", torch.tensor(ring))
        self.register_buffer("w_5", torch.ones(1, 1, 5, 5))
        rm = np.array([[min(r, c, 7 - r, 7 - c) for c in range(8)] for r in range(8)], np.float32)
        self.register_buffer("ringm", torch.tensor(rm).reshape(1, 1, 8, 8))

    def forward(self, x):
        cnt = x.sum((2, 3))                                   # (B,16)
        n_own = cnt[:, 0:5].sum(1, keepdim=True)
        n_en = cnt[:, 6:11].sum(1, keepdim=True)
        kO = cnt[:, 5:6]; kE = cnt[:, 11:12]
        al = x[:, 13, 0:1, 0]; u = x[:, 14, 0:1, 0]; mag = x[:, 15, 0:1, 0]   # (B,1)
        al1 = (al < 0.75).float()
        gate_fw = 1.0 - (u > 0.999).float() * (mag > 0.5).float()
        common = (n_own < 0.5).float() * (kO > 0.5).float() * (kE > 0.5).float() * al1 * gate_fw
        gA = common * (n_en > 0.5).float() * (n_en < 1.5).float()     # (B,1)
        gB = common * (n_en < 0.5).float()
        ownK = x[:, 5:6]; enK = x[:, 11:12]
        enNK = x[:, 6:11].sum(1, keepdim=True)
        locked = x[:, 12:13]
        empty = 1.0 - torch.clamp(x[:, 0:13].sum(1, keepdim=True), 0.0, 1.0)
        adjO = F.conv2d(ownK, self.w_adj, padding=1)
        adjE = F.conv2d(enK, self.w_adj, padding=1)
        farE = 1.0 - torch.clamp(F.conv2d(enK, self.w_5, padding=2), 0.0, 1.0)
        rE = (enK * self.ringm).sum((2, 3), keepdim=True)            # (B,1,1,1)
        better = ((self.ringm - rE + mag.reshape(-1, 1, 1, 1)) > 0.5).float()
        W = farE * better * (1.0 - locked)                             # winning landing squares
        kadj = ((adjO * enK).sum((2, 3)) > 0.5).float()                # (B,1)
        pieceAdjE = ((adjE * enNK).sum((2, 3)) > 0.5).float()
        win1 = kadj * (1.0 - pieceAdjE)
        win2 = ((adjO * enNK * W).sum((2, 3)) > 0.5).float()
        vA = LOSE_A + (1.0 - LOSE_A) * torch.clamp(win1 + win2, 0.0, 1.0)
        winB = torch.clamp(kadj + ((adjO * empty * W).sum((2, 3)) > 0.5).float(), 0.0, 1.0)
        vB = 2.0 * winB - 1.0
        gate = torch.clamp(gA + gB, 0.0, 1.0)
        v = (gA * vA + gB * vB).reshape(-1)
        bad_t = enNK * (1.0 - W) + enK * pieceAdjE.reshape(-1, 1, 1, 1)
        veto = (ownK.reshape(-1, 64, 1) * bad_t.reshape(-1, 1, 64)).reshape(-1, 4096) * (VETO * gA)
        return gate.reshape(-1), v, gA, veto


def build(base_path, out_path):
    import onnx
    from onnx import helper, TensorProto, numpy_helper, compose
    from c1 import fix_constants, validate
    tmp = Path(out_path).with_suffix(".eg_part.onnx")
    m = EGOverride().eval()
    x = torch.zeros(3, 16, 8, 8)
    torch.onnx.export(m, (x,), str(tmp), input_names=["observation"],
                      output_names=["eg_gate", "eg_value", "eg_gate_p", "eg_veto"], opset_version=17,
                      dynamic_axes={"observation": {0: "batch"}, "eg_gate": {0: "batch"}, "eg_value": {0: "batch"},
                                    "eg_gate_p": {0: "batch"}, "eg_veto": {0: "batch"}},
                      do_constant_folding=True, dynamo=False, external_data=False)
    fix_constants(tmp)
    base = onnx.load(str(base_path))
    part = onnx.load(str(tmp))
    part = compose.add_prefix(part, "egx_")
    g, pg = base.graph, part.graph
    # rewire part input to the shared observation
    for n in pg.node:
        for i, nm in enumerate(n.input):
            if nm == "egx_observation":
                n.input[i] = "observation"
    # rename base outputs
    for n in g.node:
        for i, nm in enumerate(n.output):
            if nm == "policy":
                n.output[i] = "policy_base"
            elif nm == "value":
                n.output[i] = "value_base"
    g.node.extend(pg.node)
    g.initializer.extend(pg.initializer)
    g.initializer.append(numpy_helper.from_array(np.array(0.5, np.float32), "egx_half"))
    g.node.extend([
        helper.make_node("Greater", ["egx_eg_gate", "egx_half"], ["egx_gate_b"], name="egx_gate_bool"),
        helper.make_node("Where", ["egx_gate_b", "egx_eg_value", "value_base"], ["value"], name="egx_value_where"),
        helper.make_node("Greater", ["egx_eg_gate_p", "egx_half"], ["egx_gate_pb"], name="egx_gate_pbool"),
        helper.make_node("Add", ["policy_base", "egx_eg_veto"], ["egx_policy_veto"], name="egx_policy_add"),
        helper.make_node("Where", ["egx_gate_pb", "egx_policy_veto", "policy_base"], ["policy"], name="egx_policy_where"),
    ])
    onnx.checker.check_model(base)
    onnx.save(base, str(out_path))
    tmp.unlink()
    m2 = onnx.load(str(out_path))
    print(f"nodes {len(m2.graph.node)} initializers {len(m2.graph.initializer)}")
    ok, msg = validate(out_path)
    print("validator:", "ACCEPTED" if ok else "REJECTED", [ln.strip() for ln in msg.splitlines() if "dry run" in ln])
    return ok


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
