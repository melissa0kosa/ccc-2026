"""Launcher for the residual-on-hand_v2 loop (torch-free top level for spawn workers).

Run from repo root:  uv run python levels/level4/agents/az/res.py [--resume ckpt --start-version N]
"""
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import reslib

    reslib.main()
