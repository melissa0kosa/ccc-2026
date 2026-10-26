"""cProfile of the GPU self-play driver (16 games, 64 in flight max)."""
import cProfile, pstats, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_sp as G
import torch
torch.set_num_threads(1)
ev = G.TorchEval(G.load_torch_model("rsv125"))
cfg = G.M.MCTSConfig(simulations=100, warmup_temperatures=(1.0, 1.0, 0.7, 0.5))
pr = cProfile.Profile(); t = time.time(); pr.enable(); res = G.play_many(ev, 24, 24, cfg, 5); pr.disable()
print(f"24 games {time.time()-t:.1f}s (profiled), gpu {ev.t:.1f}s")
pstats.Stats(pr).sort_stats("tottime").print_stats(16)
