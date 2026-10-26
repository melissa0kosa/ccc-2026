"""Scratch: shared loaders for the az_res_v6 review (matches 14:43:30-14:53:30)."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "reviews"))
LO, HI = "2026-10-23T14:43:30", "2026-10-23T14:53:30"
MF = HERE / "data/matches/matches_az_res_v6.json"

def load(protocol_only=True):
    ms = json.loads(MF.read_text())
    return [m for m in ms if LO <= m["played_at"] <= HI and (m.get("protocol") or not protocol_only)]
