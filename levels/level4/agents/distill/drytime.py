"""Dry-run time as the validator measures it (in-process, several repeats) + node/initializer counts."""
import sys
from pathlib import Path
import onnx
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "data" / "colony-one-env"))
from colony_one.submission_policy import inspect_and_dry_run_model  # noqa: E402
for p in sys.argv[1:]:
    m = onnx.load(p)
    ts = sorted(inspect_and_dry_run_model(Path(p)) for _ in range(5))
    print(f"{Path(p).name}: nodes {len(m.graph.node)} inits {len(m.graph.initializer)} "
          f"dry run min {ts[0]*1e3:.0f} ms median {ts[2]*1e3:.0f} ms", flush=True)
