# Gate B results vs live az_res4f_v2 (md5 f0329dfae66e629c61accd855815f2a2), 15:31-15:40 UTC

Stopped at 15:40 on coordinator request (CPU reassigned to the az_res4f_v3 / endgame-override track).
Method: `gate/garena.py`, 100 sims, alternating colours, fresh `secrets` seeds, 4 workers.
Logs: `gate/gateB_games1.jsonl` (screen, 600 games) and `gate/gateB_games2.jsonl` (tta2 extension, killed at 112/180; the greedy and stopgap runs never started).
All games are distinct: maxdup 1 per pairing, and 230/231 post-warmup boards are distinct in the pooled tta2 set.

| candidate | md5 | vs az_res4f_v2 (Wilson 95%) | cyan / magenta | cpu/game (both sides) | verdict |
|---|---|---|---|---|---|
| prior_4f2_v0 | 4af71004333073ed4e37af642d332363 | 64/120 = 53.3% [44.4, 62.0] | 58% / 48% | 1.84 s | FAIL (screen < 55%) |
| deriv_4f2_tta2 | 0e2cd4b4ee0add436c4b9e5967f5780c | screen 67/120 = 55.8%; extended to **131/232 = 56.5% [50.0, 62.7]**; with the deriv agent's 52/100: 183/332 = 55.1% [49.7, 60.4] | 61% / 52% | 2.36 s | FAIL / not passable: lower bound 0.50, far below 0.55 (0.60 after 15:50) |
| deriv_4f2_tta4 | 864692b721878ee589d651855b50aa52 | 56/120 = 46.7% [38.0, 55.6]; with the deriv agent's 55/100: 111/220 = 50.5% [43.9, 57.0] | 47% / 47% | 3.39 s | FAIL; tta8 skipped as instructed |
| distill_v6hf | 6dac817f946428f85231e24b782baac3 | 54/120 = 45.0% [36.4, 53.9] | 42% / 48% | 2.11 s | FAIL |
| distill_v6s8 | 1ff804d7389f50ab1a19a8d844f9774e | 50/120 = 41.7% [33.2, 50.6] | 38% / 45% | 1.84 s | FAIL |

No candidate reached 60% over at least 120 games. The best was tta2 at 56.5% over 232 games.

## TTA exactness (`gate/gateB_tta.py`, 96 real positions from greedy games)
- **tta2 (syms {0, 1}, column mirror):**
  - Matches the numpy average of az_res4f_v2 over the group: max |dp| 0, max |dv| 1.8e-7.
  - Exactly equivariant under sym 1: dp 0, dv 0.
  - Outside the group (syms 2-7), dp is 2.1-2.6 and dv 0.12-0.19. This is expected: the base network is not D4-symmetric.
- **tta4 (syms {0, 1, 2, 3}):**
  - Matches the reference: dp 0, dv 1.8e-7.
  - Equivariant under syms 1-3: dp ≤ 1.9e-6, dv ≤ 1.8e-7.
  - Outside the group (syms 4-7), dp is 1.55.
- Legal-move D4 symmetry was already validated by `agents/az/symcheck.py` (0 mismatches).
- **TTA vs base:** TTA changes the network a lot (tta2 max |dp| 1.4, max |dv| 0.12 vs the base). It is a different player, not a refinement.

## CPU
- **Evaluator latency (referee OnnxEvaluator, interleaved):**

  | model | b16 | b8 | b1 |
  |---|---|---|---|
  | az_res4f_v2 | 4.68 ms | 2.81 ms | 1.03 ms |
  | tta2 | 9.70 ms (2.07x) | 4.84 ms (1.72x) | 1.30 ms (1.26x) |
  | tta4 | 20.99 ms (4.49x) | 9.58 ms (3.41x) | 1.75 ms (1.70x) |

- **Our side per game,** estimated as game CPU minus about 0.64 s for the base side (base vs base is about 1.28 s per game). The direct per-side script `gate/gateB_cpu.py` was written but not run, because the CPU was reassigned.
  - tta2: about 1.7 s per game, about 75 ms per action. A 52-action side (104-action game) comes to about 3.9 s.
  - tta4: about 2.75 s per game, about 115 ms per action, so about 6 s for 52 actions.
  - Both are under the 15 s RLIMIT_CPU (13.5 s validator approximation) on this loaded box. tta4 has about 2x headroom only if the server is no more than about 2x slower.
- Validator, greedy and stopgap gates were not run for any candidate: none passed the vs-seat criterion.

**Verdict: no upload from gate B.**
