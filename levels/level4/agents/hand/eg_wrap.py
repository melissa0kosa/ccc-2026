"""Add the exact gated bare-Kernel endgame override (value + prior veto) to ANY policy/value ONNX model.

usage (repo root): uv run python levels/level4/agents/hand/eg_wrap.py <src.onnx> <dst.onnx> [n_brute=600]
Steps: graph surgery (eg_override.build) -> official validator -> parity vs src on 6000 real self-play
positions (must be bit-identical outside the gate) -> brute force vs the env on generated gated positions.
Prints md5 and a one-line verdict; exit code 1 on any failure.
"""
import hashlib, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import numpy as np


def main():
    src, dst = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    import eg_override, eg_verify
    ok_val = eg_override.build(src, dst)
    base, eg = eg_verify.sess(src), eg_verify.sess(dst)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eg_verify.parity(base, eg)
    par = buf.getvalue().strip(); print(par)
    sys.argv = [sys.argv[0], src, dst, str(n)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eg_verify.main()
    out = buf.getvalue().strip().splitlines()
    bf = [ln for ln in out if ln.startswith("[brute force]")][0]
    print(bf)
    ok_par = "value bit-identical outside gate: True" in par and "policy bit-identical outside gate: True" in par
    ok_bf = "'bad_v': 0," in bf and "'bad_veto': 0," in bf and "'bad_off': 0," in bf
    md5 = hashlib.md5(Path(dst).read_bytes()).hexdigest()
    verdict = "PASS" if (ok_val and ok_par and ok_bf) else "FAIL"
    print(f"{verdict} {dst} md5 {md5} (validator {'ACCEPTED' if ok_val else 'REJECTED'}, parity {ok_par}, brute force {ok_bf})")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
