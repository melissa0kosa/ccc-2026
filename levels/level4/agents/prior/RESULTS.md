# prior agent: results

## STATE (15:41 UTC): STOPPED on coordinator request
All jobs are stopped and none of my processes are running. No upload decisions were made here.
- The coordinator reports that prior_4f2_v0 screened at 37/78 vs az_res4f_v2 on their side. My own runs (below) gave 168/300 = 56%. The two results disagree, so treat the R1x3 effect on az_res4f_v2 as unproven.
- Δp distillation on az_res4f_v2 (prior_4f2_v1w) reached 189/300 = 63% vs az_res4f_v2 but only 103/200 = 52% [45%,58%] head-to-head vs prior_4f2_v0. It is not flagged.

## A. Warmup sharpening wrapper on az_res_v20: `levels/level4/models/az_res_v20_warm.onnx` (az_res_v20 + round-1 prior x3, value untouched)
- vs az_res_v20 (100 sims, colours alternate, `c1.arena`): run 1 122/200 = 61% [54%,67%] (cyan 66% / magenta 56%); run 2 (new seed) 117/200 = 58% [52%,65%] (cyan 61% / magenta 56%). Pooled 239/400 = 59.8%, Wilson 95% [54.9%, 64.4%].
- Validator ACCEPTED (70.6 ms dry run). Earlier "too slow to check" failures also hit the unmodified champion at the same moment, i.e. machine load.
- 215 nodes / 264 initializers (champion 204 / 258). CPU 1.3-1.5 s per game for both sides together.

`warm.py` edits the ONNX graph directly; the value output is untouched:

    policy_out = policy * (1 + a*r1 + b*r2)
    r1 = |plane14 - 0.125| < 0.01 AND sum(plane12) < 0.5   (round 1)
    r2 = |plane14 - 0.25 | < 0.01 AND sum(plane12) < 0.5   (round 2)

- Rounds 9, 10, 17 and 18 have the same urgency values but locked squares, so the factor stays 1 there.
- Added ops: Flatten, MatMul x3, Sub, Abs, Less, Cast, Mul x2, Add (11 nodes, 6 initializers).
- Checked on 1530 positions: value bit-identical; policy = champion x3 in round 1 only.
- Usage: `uv run python levels/level4/agents/prior/warm.py <src.onnx> <dst.onnx> 2.0 0.0`

Offline Q-loss (`warm_eval.py`, `warm_eval2.py`):
- Positions: 175 round-1 and 175 round-2 positions from hand_v3 self-play played under the referee config, 10 seeds each, same seeds across variants.
- Reference: az_res_v20 800-sim search with no noise.
- Score: Q800(best) - Q800(sampled move), where each variant samples under the referee config (100 sims, noise, warmup temperature).

| factor (round 1) | round-1 Q-loss | d vs champion (paired) | best-move rate | loss > 0.2 |
|---|---|---|---|---|
| x1 (champion) | 0.119 | - | 32.6% | 391/1750 |
| x1.5 | 0.102 | -0.017 +- 0.003 | 35.8% | 326 |
| x2 | 0.094 | -0.025 +- 0.004 | 37.7% | 293 |
| **x3 (chosen)** | **0.087** | **-0.032 +- 0.004** | 40.5% | 260 |
| x4 | 0.085 | -0.034 +- 0.004 | 41.4% | 254 |
| x6 | 0.084 | -0.036 +- 0.004 | 42.1% | 241 |

- Round 2 showed no gain at x1.5 or x2 (+0.002 / +0.000 +- 0.002), so b = 0.
- The review's `hv_scan2.json` has no observations, so live positions could not be reused.

## B. Δp on az_res4f_v2 (`common_4f.py: Prior4f`)
- Model: the champion's value and logits are frozen. A zero-initialised Δp comes from a trunk over [obs, champion trunk h]: 24 channels, 2 residual blocks, a 1x1 conv to 64 channels (from-square) plus a rank-8 bilinear term. A learned temperature is applied to the champion logits.
- Data: az_res4f_v2 self-play, every position labelled with an 800-sim search: 69 games, 2.76k positions (`data/gen_r4f_*.pkl`).
- Held-out (7 games, 308 positions): CE 1.877 -> 1.799. Top-1 agreement with the 800-sim best 0.649 -> 0.627 (warmup 0.607 -> 0.679, post-warmup 0.659 -> 0.615). So Δp mostly sharpens the opening prior.
- `prior_4f2_v0` (R1x3 only, 289 nodes / 327 initializers, ACCEPTED) vs az_res4f_v2: 60/100 + 108/200 = 168/300 = 56.0% [50.3%, 61.4%] (cyan 62%/60%, magenta 58%/48%).
- `prior_4f2_v1w` (Δp + R1x3, 315 nodes / 347 initializers; validator not yet passed, "too slow" under load) vs az_res4f_v2: 58/100 + 131/200 = 189/300 = 63.0% [57.4%, 68.2%] (cyan 66%/68%, magenta 50%/63%). Head-to-head vs prior_4f2_v0: 103/200 = 52% [45%,58%].
- `prior_4f2_v1` = Δp without the wrapper; not arena-tested.

## C. Δp on hand_v3 (superseded)
- Data: 8.2k positions of hand_v3 self-play with 800-sim labels.
- Result: held-out top-1 agreement 0.364 -> 0.559 (warmup 0.301 -> 0.659).
- Not arena-tested; the base is only hand_v2 strength.
- D4 augmentation was verified first: 0 legal-move mismatches in 2640 checks.
