# Upload gate (live seat az_res_v20.onnx since 14:53 UTC)

## 15:18 UTC: RELEASE az_res4f_v2.onnx now (md5 f0329dfae66e, file of 15:15:22 UTC); passes every criterion vs the live az_res_v20

| criterion | measured | result |
|---|---|---|
| Official validator | ACCEPTED on the 1st try, 102.5 ms dry run | PASS |
| Graph caps | 278 nodes / 321 initializers | PASS |
| >= 200 games vs az_res_v20, Wilson lower > 0.55 | **144/220 = 65.5% [59.0, 71.4]** | PASS |
| Effective sample | 220/220 distinct games, 217/219 distinct post-warmup boards | PASS |
| Colours | cyan 69% (110) / magenta 62% (110) | PASS |
| vs greedy | 20/20 | PASS |
| vs stopgap_v1 | 58/60 = 96.7% (seat 60/60, -3.3 pp) | PASS |
| Old-seat guard vs hand_v2 | 69/80 = 86.2% (seat az_res_v20: 49/60 = 81.7%) | PASS |
| CPU per game, both sides | 1.28 s | PASS |

Margin: about +110 Elo locally vs the live seat, so worth about 2 forfeits plus swing with about 1.5 h left (an ordinary upload, before the 15:50 cutoff). Caveat: it is a sibling of v20 (same az residual family, hand_v4f base), and head-to-heads have been non-transitive (distill_v3: 76% vs hand_v2 but only 60% vs az_res_v6), so the live gain may be smaller.


## Round 3: candidates vs live az_res_v20 (updated 15:17 UTC)

Seat references:
- az_res_v20 vs stopgap_v1: 60/60
- az_res_v20 vs greedy: 77/80 (20/20 + 57/60)
- az_res_v20 vs hand_v2: 49/60 = 81.7% (sanity)

| candidate | vs az_res_v20 (Wilson 95%) | cyan / magenta | distinct games | greedy | stopgap_v1 | cpu/game | validator / caps | gate |
|---|---|---|---|---|---|---|---|---|
| az_res_v20_warm (md5 6b06069d, file of 15:05:54) | own 318/560 = 56.8% [52.7, 60.8]; pooled with the prior agent's 239/400 (same c1.arena method): 557/960 = 58.0% [54.87, 61.1] | 54% / 60% | 557/560 games; post-warmup boards 531/556 (n_eff about 500) | 95/98 = 96.9% | 57/60 = 95.0% (-5 pp) | 1.38 s | ACCEPTED (94.2 ms) / 215, 264 | **FAIL**: lower bound 0.527 own, 0.549 pooled, both < 0.55. A probable small edge (+50 Elo locally) that does not justify about 2 forfeits plus swing |
| az_res4f_v1 | 119/220 = 54.1% [47.5, 60.5] | 52% / 56% | 220/220 | 20/20 | 16/17 | 1.94 s | load-rejected 3x / 278, 321 | **FAIL** |
| distill_v5s8 | 110/220 = 50.0% [43.4, 56.6] | 53% / 47% | 220/220 | 11/11 | - | 2.10 s | load-rejected 3x / 134, 126 | **FAIL** |
| distill_v5hf | 57/120 = 47.5% [38.8, 56.4] | 57% / 38% | 120/120 | - | - | 1.76 s | ACCEPTED (150 ms) / 285, 327 | **FAIL** (futility stop) |
| distill_v4s8 | 62/147 = 42.2% [34.5, 50.3] | 41% / 44% | 147/147 | - | - | 2.72 s | ACCEPTED (50.8 ms) / 134, 126 | **FAIL** (futility stop) |
| az_res_v31 | 58/120 = 48.3% [39.6, 57.2] | 42% / 55% | 120/120 | - | - | 1.99 s | - | **FAIL** (screen; the az agent also has v30 53%, v31 46%, v32 48%) |
| az_res_ens2 | 57/120 = 47.5% [38.8, 56.4] (screen) | 57% / 38% | 120/120 | - | - | 1.72 s | - / 239, 290 | **FAIL** |
| az_res4f_v2 (md5 f0329dfa) | 144/220 = 65.5% [59.0, 71.4] | 69% / 62% | 220/220 | 20/20 | 58/60 | 1.28 s | ACCEPTED (102.5 ms) / 278, 321 | **PASS: upload** |
| hand_v4f (reference) | 47/200 = 23.5% [18.2, 29.8] | 22% / 25% | 200/200 | | | 1.35 s | | reference only |
| distill_v4hb | not run (47/100 vs distill_v4s8 per the distill agent) | | | | | | 211 / 264 | skip |

Validator note:
- From about 14:58 the box is at 100% CPU (about 32 python processes). The official validator then rejects everything with "too slow to check", including the live az_res_v20 (control run at 14:58).
- Relative in-process latency (batch 8, referee evaluator) vs az_res_v20 = 1.00: warm 1.08, distill_v5s8 1.11, az_res4f_v1 1.31, distill_v5hf 1.46.
- Re-validate on a quieter moment before any upload.

UPLOAD: az_res4f_v2.onnx

---
## Round 2 (seat az_res_v6, 14:43-14:53): az_res_v20 released at 14:52

| criterion | measured | result |
|---|---|---|
| Official validator | ACCEPTED on the 1st try, 96.0 ms dry run | PASS |
| Graph caps | 204 nodes / 258 initializers | PASS |
| >= 200 games vs az_res_v6, Wilson lower > 0.55 | **170/220 = 77.3% [71.3, 82.3]**; lower bound > 0.60, so a big edge | PASS |
| Effective sample | 220/220 distinct games, 216/217 distinct post-warmup boards, dedup Wilson [71.3, 82.3] | PASS |
| Colours | cyan 77% (110) / magenta 77% (110) | PASS |
| vs greedy | 20/20 | PASS |
| vs stopgap_v1 (no gauntlet drop) | 60/60 = 100% (az_res_v6 56/60 = 93.3%, hand_v2 75/98 = 76.5%) | PASS |
| CPU per game, both sides | 2.28 s (vs az_res_v6), 44 actions | PASS |

Worth the swap cost (~2 forfeits + ~10 games of swing): yes. A 77% head-to-head rate against the live seat is about +210 Elo locally.

Other results vs az_res_v6:
- distill_v3: 120/200 = 60.0% [53.1, 66.5], cyan 59% / magenta 61%, 2.66 s. Below the bar, and v20 is stronger anyway.
- distill_v3hb: 14/30, stopped (skipped per coordinator).

Running now (4 workers, games9.jsonl): distill_v4s8 vs az_res_v20 (240), distill_v4s8 greedy/stopgap, az_res_v20 vs hand_v2 (60, sanity).
distill_v4s8: validator ACCEPTED on the 1st try (50.8 ms), 134 nodes / 126 initializers.

---
## Earlier round (seat hand_v2), kept for the record

**Status 14:43 UTC: RELEASE az_res_v6.onnx now.** Under the new policy ("release as soon as a candidate passes all gates") it qualifies: it is the first candidate to pass every criterion.
Still running in the background (4 workers): distill_v3 vs hand_v2 (146/190 = 76.8% so far), distill_v3hb vs hand_v2 (39/56 = 69.6% so far) plus its greedy/stopgap gates, then distill_v3 vs az_res_v6 head-to-head (200). These decide whether a second upload is worth its ~2 forfeits.

All numbers: 100 sims, alternating colours, fresh `secrets` seeds, games with the c1-identical loop
(`gate/garena.py`, per-game JSONL with action-sequence hashes in `gate/games*.jsonl`), tables generated by `gate/report.py`.
Baseline: hand_v2 vs stopgap_v1 = 75/98 = 76.5% (two fresh-seed runs: 49/60 and 26/38).

| candidate | vs hand_v2 (Wilson 95%) | cyan / magenta | greedy | vs stopgap_v1 (delta vs hand_v2's 76.5%) | cpu/game | validator / caps | gate |
|---|---|---|---|---|---|---|---|
| **az_res_v6** | **138/200 = 69.0% [62.3, 75.0]** | 70% / 68% | 20/20 | 56/60 = 93.3% (+16.8 pp) | 1.98 s | ACCEPTED try 1 / 503 nodes, 444 inits | **PASS** |
| distill_v3 | 132/172 = 76.7% [69.9, 82.4] (running to 200) | 73% / 80% | pending | pending | 1.85 s | ACCEPTED try 1 / 134, 126 | pending |
| distill_v3hb | just started | | | | | ACCEPTED try 1 (144 ms dry run) / 241, 321 | pending |
| distill_v2 | 129/200 = 64.5% [57.7, 70.8] | 61% / 68% | pending (lower priority) | 49/54 = 90.7% (+14.2 pp) | 2.13 s | ACCEPTED / 134, 126 | vs-seat PASS, greedy pending |
| hand_v3 | 120/244 = 49.2% [43.0, 55.4] | 52% / 46% | - | - | 1.35 s | ACCEPTED / 167, 225 | **FAIL** |
| distill_v2n | 102/200 = 51.0% [44.1, 57.8] | 43% / 59% | - | - | 1.76 s | ACCEPTED / 134, 126 | **FAIL** |
| champion.onnx | byte-identical to hand_v2 (same md5) | | | | | | n/a |

## hand_v3: why the hand agent's 69% is wrong and the 50% is right
- **The scripts are mechanically identical.** `agents/hand/myarena.py` calls `c1.make_agent` with the same seeds (seed+2g / seed+2g+1), the same g%2 colour alternation, 100 sims and the same referee MCTS. Only its logging differs.
- **Independent fresh-seed measurements all say about 50%:**
  - main session: 40/80
  - gate: 120/244 vs hand_v2
  - gate: 70/120 vs hand_v2fast
  - hand_v2fast is output-identical to hand_v2 on 1,962 game positions (max |dv| 3e-7, max |dp| 1.5e-7, prior argmax never differs), so that run samples the same matchup.
  - Pooled: 230/444 = 51.8% [47.2, 56.4].
- **83/120 is not seed luck.** Under p = 0.52 its probability is about 1e-4 (z ≈ 3.8). Nor is it a small effective sample size (see below). It must come from a different experiment: most likely a pre-final export of `hand_v3.onnx` that the later export (14:13:17 UTC, after `w_v3.json`, "+ Cyan-specific firewall terms") overwrote, or selective reporting across reruns.
- **The hand agent's own follow-ups also point to about 50%.** v3a/b/c and v4a/b vs v3 came out at 42–52%.

## Effective sample size
- The referee is greedy after 8 noisy opening actions, so I hashed every game's full action sequence and the board after action 8.
- 239/240 (hand_v3), 200/200 (distill_v2, az_res_v6) and 172/172 (distill_v3) games are distinct. 198/198 and 170/172 post-warmup boards are distinct.
- So duplication does not shrink n (n_eff ≈ n), and the Wilson CIs above stand.
- **Caveat:** distinct is not the same as independent of the opponent pool. All games are against one local opponent; the live pool differs, and a deterministic model repeats its systematic mistakes in every game that reaches them. Treat the vs-hand_v2 margin as an upper bound on the live Elo gain, not a forecast.
- Also, 40–60-game arenas carry ±12–15 pp of noise: az's identical-model sanity runs gave 35% and 45%.

(14:43 verdict, deployed: az_res_v6.onnx)

(14:52 verdict, deployed 14:53: az_res_v20.onnx)

UPLOAD: az_res4f_v2.onnx
