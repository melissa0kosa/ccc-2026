"""Launcher (torch-free top level so spawned workers stay light).

Run from repo root:  uv run python levels/level4/agents/az/az.py [--resume ckpt] [--load-data] ...
"""
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import azlib

    azlib.main()
