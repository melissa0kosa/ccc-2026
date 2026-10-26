"""Make a weight variant: variant.py base.json out_name soft=0.5 hard=win_now:3,win2:3 bias=0 pscale=1.0 [hardscale=1]
Exports levels/level4/models/hand_<out_name>.onnx and data/w_<out_name>.json"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
HARD = ["win_now", "win2", "enK_N1_u1_mag"]
if __name__ == "__main__":
    base = json.loads(Path(sys.argv[1]).read_text()); name = sys.argv[2]
    kw = dict(a.split("=", 1) for a in sys.argv[3:])
    soft = float(kw.get("soft", 1)); pscale = float(kw.get("pscale", 1))
    w = {k: (v if k in HARD else v * soft) for k, v in base["weights"].items()}
    for item in filter(None, kw.get("hard", "").split(",")):
        k, v = item.split(":"); w[k] = float(v)
    bias = float(kw["bias"]) if "bias" in kw else base.get("bias", 0.0) * soft
    pol = dict(base.get("pol") or {})
    for item in filter(None, kw.get("pol", "").split(",")):
        k, v = item.split(":"); pol[k] = float(v)
    for k in ("k_d2", "k_esc_thr", "guard", "tie", "esc", "esc_def", "fw_from", "k_from", "cap", "cap_hang", "near_enK", "fw_to", "center", "danger", "danger_def", "k_fw"):
        if k in pol:
            pol[k] *= pscale
    out = {"weights": w, "bias": bias, "pol": pol}
    p = Path(__file__).resolve().parent / "data" / f"w_{name}.json"
    p.write_text(json.dumps(out, indent=1))
    import export_model
    export_model.export(name, w, bias, pol)
