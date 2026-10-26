# GPU-batched self-play (Level 4, COLONY: ONE)

Status at 16:20 UTC: a working GPU-batched self-play generator, done, with no processes left running. It has produced 1,712 games (79k positions). It produces valid games in the exact
`sp_v4f.pkl` record format, and the az loader (`reslib.load_hand_sp`) reads them unchanged.
Throughput per CPU process is about **2x** the CPU/ONNX pipeline when 200-256 games run in flight, and about the same
with only 24-32 in flight. The remaining wall is the pure-Python rules engine, not the network.

## Repo choice: none, we wrote our own (about 200 lines)

GitHub search (about 5 min) plus what is known about the usual candidates:

| Repo | Licence | Windows | Real cross-game GPU batching | Plug-in effort | Verdict |
|---|---|---|---|---|---|
| suragnair/alpha-zero-general | MIT | yes | **no**: one tree, one leaf at a time | low, but its MCTS flips the value sign every ply (`v = -search(...)`), which is wrong for 2 actions per turn | no gain |
| werner-duvaud/muzero-general | MIT | Ray on Windows is shaky | no (Ray actors, one game each) | high, and it is MuZero (learned dynamics), not the referee search | no |
| opendilab/LightZero | Apache-2.0 | ctree (C++/Cython) build is Linux-first | yes (C++ batched MCTS over envs) | high: DI-engine stack plus a C++ build, and our engine stays in Python anyway | post-contest only, if the rules move to C++ |
| google-deepmind/mctx | Apache-2.0 | JAX GPU not native on Windows | yes | needs the game in JAX | out |
| MiniZero / PolyEnv MctsPool / kzero | various | C++/Rust | yes | need a native rules port | post-contest idea only |

Why we wrote our own: every repo above batches the **network**. After batching, our cost sits in the
**Python engine** (`legal_actions`, `encode_board`, `_set_priors`, `_select_action`), and no repo changes that without
a rules re-implementation. The referee's own `colony_one/mcts.py` already separates "descend and collect" from "evaluate",
so we turned `mcts.run` into a generator and ran 200+ games in one process. The search semantics are
**exactly** the referee's: PUCT 1.5, FPU 0, batch 8 plus virtual loss, 2 actions per turn, the sign flips only at a turn change,
and the warmup noise and temperature are the same.

`git clone --depth 1` of alpha-zero-general into `levels/level4/ext/` failed twice (`fetch-pack: invalid index-pack output`, network).
It wasn't needed, so `ext/` is empty. Nothing external was run, and no packages were added.

## Files (all in `levels/level4/agents/gpu/`)

- `gpu_sp.py`: the generator.
  - `load_torch_model("rsv125"|"v2"|path)` loads `agents/az/ckpt_res/az_res4f_v2.pt` (read-only) into `reslib.ResFast(32,2,kind="v4f")`. `rsv125` is v2 with `vfc2` scaled by 1.25, the same as `deriv/derive.py`'s dv×1.25.
  - `TorchEval` runs fp32 on CUDA with TF32 off and in chunks of 2048.
  - `search_gen` is `mcts.run` line for line with `yield planes`. It reuses the referee internals `_descend/_set_priors/_backup`.
  - `game_gen` plays one game. `play_many` keeps `--par` games in flight and makes one GPU forward per round over every pending leaf of every game.
  - Run: `uv run python levels/level4/agents/gpu/gpu_sp.py --games 400 --par 200 --tag g1 [--model rsv125] [--warm 1.0,1.0,0.7,0.5]`. In PyCharm, open the file and press Run. The defaults are a 64-game test.
  - The warmup default is gen_data.py's: temperatures 1.0, 1.0, 0.7, 0.5 for 4 actions each, with root noise for the same 16 actions, then greedy play. The current version saves partial data every 30 s.
  - Output: `data/gsp_<model>_<tag>.pkl` (records) plus `.meta.pkl` (actions per game, winner, seed, settings).
- `check.py`: validity checks (see below). `uv run python levels/level4/agents/gpu/check.py data/gsp_x.pkl`
- `prof_cpu.py`, `prof_gpu.py`: profiling of the CPU referee pipeline and the GPU driver.

## Validity checks (all pass)

1. Torch model vs deployed `models/deriv_4f2_rsv125.onnx` on 200 real positions: max |Δlogit| 1.2e-5, max |Δv| 8e-7.
2. Generator search vs referee `mcts.run`, same ONNX evaluator and the same seed: **identical root visit counts on 12/12** positions.
3. Replay of `gsp_rsv125_smoke.pkl` through the real env (32 games, 1,424 records):
   - every stored observation equals the env's;
   - every chosen action is legal and appears in the stored visit indices;
   - every game ends terminal with the stored winner;
   - 0 bad records.
4. `reslib.load_hand_sp` loads `gsp_rsv125_p256.pkl`: 11,793 positions, mean z 0.01, every policy sums to 1, q in [-1, 1].

## Throughput (same machine; other agents kept the CPU at about 82% load and the GPU busy with az training)

| Setup | games/min | positions/min | Notes |
|---|---|---|---|
| CPU referee, ONNX batch 8, 1 process, lighter load (15:53) | 41 | 2,420 | NN = 64% of wall time |
| CPU referee, ONNX batch 8, 1 process, 82% load (16:06) | 19 | 1,126 | NN = 65% |
| GPU, 1 process, 24 in flight (16:07, loaded) | 34 | 1,323 | GPU 47% of wall (73 pos/call, about 16 ms fixed per forward) |
| GPU, 1 process, 32 in flight | 40 | 1,795 | |
| GPU, 1 process, 256 in flight | **50** | **2,300** | GPU only 16% of wall (812 pos/call); peak VRAM 495 MiB |
| GPU, 3 processes × 200 in flight (16:08-16:19, loaded) | **107** total (36 each) | **4,920** total (1,640 each) | 1,200 games in about 11 min; GPU 69 s of 660 s per process; peak VRAM 400 MiB each |
| (CPU referee, 3 workers, estimate) | 57-123 | 3,400-7,300 | 3 × the single-process rows above, from 82% load to light load |

Conclusions:
- Batching removes the network cost, which is about 64% of the CPU pipeline. The Python engine (about 0.01-0.02 s per searched
  position under load) is now about 84% of wall time, so the per-process gain caps near 2-2.5x.
- With fewer than about 64 games in flight, the fast4 hand-feature base's fixed cost of about 16 ms per forward (many small kernels) eats the gain.
- VRAM: about 0.5 GB peak per process plus the CUDA context, so 3 processes stay under 4 GB.

## Data produced

- `data/gsp_rsv125_smoke.pkl`: 32 games, 1,424 positions (replay-validated).
- `data/gsp_rsv125_p256.pkl`: 256 games, 11,793 positions (loader-validated).
- `data/gsp_rsv125_cmp.pkl`: 24 games, 922 positions.
- `data/gsp_rsv125_g1.pkl` (17,788 positions, 18.5 MB), `g2` (18,142 positions), `g3` (18,755 positions): 3 × 400 games, rsv125 self-play,
  100 sims, cyan wins 53%, average length 45.6 actions. g1 is fully replay-validated: 400/400 games, 0 bad records.
- **Total: about 1,712 games and 79k positions** (sp_v4f.pkl has 30k), all from the live model deriv_4f2_rsv125.

Honest verdict: it works and the data is valid. Under today's load, the speed-up is about **1.5-2x per CPU core** over the
ONNX/CPU pipeline, not 10x, because the Python rules engine is the bottleneck.

To use it in az training, add `buf.add(load_hand_sp(Path(".../agents/gpu/data/gsp_rsv125_g1.pkl")))` next to the `sp_v4f.pkl` line
in `retrain_fast.py`, or pass it as `AZ_TEACHER`.

## Next steps (after the contest)

1. **Faster engine, the only big lever.** In the GPU driver profile:
   - `legal_actions` is about 26% (`is_locked`/`in_bounds` alone are 15M calls per 24 games);
   - `encode_board` about 16%;
   - `_set_priors` about 10%;
   - `_select_action` about 7%.

   Options, all checked by exact-equality tests against `colony_one`, like check.py [2] and [3]:
   - (a) Bitboard rules in Rust or C++ (PyO3/pybind11) exposing `legal`, `apply`, `undo` and `encode`. Expect 20-50x on the engine.
   - (b) A vectorized torch/NumPy engine over N boards on the GPU. Harder, because of the bodyguard, firewall and 2-action rules.
   - (c) Cheap semantic-preserving wins first:
     - vectorized `encode_board` from the piece list;
     - a precomputed locked-square mask instead of `is_locked` per square;
     - `_set_priors` gathering float32 legal logits before converting to float64;
     - per-node NumPy arrays for P/N/W in `_select_action`.
2. Run 3-6 processes × 256 in flight, and overlap GPU and CPU by double-buffering two halves of the pool (up to +16%).
3. Speed up the fixed forward cost: CUDA graphs for fixed batch buckets (pad to 256/512/1024/2048), or move the hand base into one fused module.
4. Once the engine is native, LightZero's ctree or a small C++ MCTS becomes worth it, because tree ops are the next Python cost.
   Keep `check.py`'s "identical visits vs referee" test as the gate for any change.
5. Training loop: generate with the current champion → `retrain_fast.py` (sp_v4f + gsp_* data) → gate in `gate/garena.py`.
   At 2,300 pos/min per process, 3 processes give about 400k positions per hour (sp_v4f.pkl is 30k).
