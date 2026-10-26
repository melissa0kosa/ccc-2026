"""Scratch: loader for the az_res4f_v6 review (played_at >= 15:57:30)."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "reviews"))
LO = "2026-10-23T15:57:30"
FA = HERE / "data/matches/matches_v6_a.json"
MODEL = "az_res4f_v6"

def load(protocol_only=True):
    a = [m for m in json.loads(FA.read_text()) if m["played_at"] >= LO]
    return sorted([m for m in a if m.get("protocol") or not protocol_only], key=lambda m: m["played_at"])
