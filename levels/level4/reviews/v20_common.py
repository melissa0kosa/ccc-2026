"""Scratch: loaders for the az_res_v20 review (played_at >= 14:53:30). Fills missing protocols from the _a download."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "reviews"))
LO = "2026-10-23T14:53:30"
FB = HERE / "data/matches/matches_az_res_v20_b.json"
FA = HERE / "data/matches/matches_az_res_v20_a.json"

def load(protocol_only=True):
    b = [m for m in json.loads(FB.read_text()) if m["played_at"] >= LO]
    a = {m["match_id"]: m for m in json.loads(FA.read_text())}
    for m in b:
        if not m.get("protocol") and a.get(m["match_id"], {}).get("protocol"):
            m["protocol"] = a[m["match_id"]]["protocol"]; m["_proto_from"] = "a"
    return sorted([m for m in b if m.get("protocol") or not protocol_only], key=lambda m: m["played_at"])
