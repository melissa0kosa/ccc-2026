"""`uv run ccc ...` helpers.

    uv run ccc new 3                   -> create levels/level3/ from the template
    uv run ccc unzip level_3.zip 3     -> extract a zip into levels/level3/data/
    uv run ccc check levels/level3/out/level_3_sample.out levels/level3/data/level_3_sample.out
    uv run ccc gpu                     -> show torch / CUDA status
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "levels" / "_template"


def cmd_new(args) -> None:
    dest = ROOT / "levels" / f"level{args.level}"
    if dest.exists():
        sys.exit(f"{dest} already exists")
    shutil.copytree(TEMPLATE, dest)
    (dest / "data").mkdir(exist_ok=True)
    (dest / "out").mkdir(exist_ok=True)
    print(f"created {dest}")


def cmd_unzip(args) -> None:
    from ccc_2026.io import extract_zip

    dest = ROOT / "levels" / f"level{args.level}" / "data"
    out = extract_zip(args.zip, dest)
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"{p.relative_to(ROOT)}  ({p.stat().st_size:,} bytes)")


def cmd_check(args) -> None:
    from ccc_2026.io import compare_to_sample

    problems = compare_to_sample(args.produced, args.expected, tol=args.tol)
    if problems:
        print("MISMATCH")
        print("\n".join(problems))
        sys.exit(1)
    print("OK - matches sample")


def cmd_gpu(_args) -> None:
    import torch

    print("torch", torch.__version__, "| cuda build", torch.version.cuda)
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0),
              "| capability", torch.cuda.get_device_capability(0))
        print("arch list:", torch.cuda.get_arch_list())
    else:
        print("CUDA not available, running on CPU")


def main() -> None:
    p = argparse.ArgumentParser(prog="ccc")
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("new", help="create levels/levelN from the template")
    s.add_argument("level")
    s.set_defaults(func=cmd_new)
    s = sub.add_parser("unzip", help="extract a level zip into levels/levelN/data")
    s.add_argument("zip")
    s.add_argument("level")
    s.set_defaults(func=cmd_unzip)
    s = sub.add_parser("check", help="compare an output file against an expected sample output")
    s.add_argument("produced")
    s.add_argument("expected")
    s.add_argument("--tol", type=float, default=1e-6)
    s.set_defaults(func=cmd_check)
    s = sub.add_parser("gpu", help="show torch / CUDA status")
    s.set_defaults(func=cmd_gpu)
    args = p.parse_args()
    args.func(args)
