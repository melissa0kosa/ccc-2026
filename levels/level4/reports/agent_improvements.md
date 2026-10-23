# Agent improvements: audit of the training pipeline (S1-S3, az, gpu)

Side session -45, 16:08-16:20 UTC, 2026-10-23. Read-only code review plus two 1-core data checks.
Builds on `reports/perfect_play_ceiling.md` (oracle hit rates, search scaling).

## Bottom line

- **No correctness bug found** in the paths that could cap all agents:
  - D4 transforms;
  - the value sign on 2-action turns;
  - canonical indices;
  - live-imitation perspective;
  - obs scaling;
  - BN recompute.
  Details in §2.
- **The binding limits are knobs, not bugs:**
  1. **Policy targets come from 100-sim searches.** About 74% of v6's data (230k positions) is self-play whose policy target is the 100-sim visit distribution. The ceiling study shows the 100-sim choice matches the deep-search move only 47% of the time, against 68% for the raw prior's top-1. So most of the policy loss pulls the move ordering toward a WEAKER operator than the net's own prior. This is the structural reason policy strength stalls while value-side changes (endgame data) keep paying.
  2. **Passes over the data, with no held-out split.** v6 = 2,626 × 2048 ≈ 17 passes over 312k (+20 pp over v4). v7 = 15k steps ≈ 98 passes, and it measured 54/100 vs v6. The replicated small sets (teacher ×3/×5, endgame `rep`) are seen hundreds of times. No trainer holds out games, so this can't be seen during training.
  3. **Raw-visit CE can't create the +2-logit margin** that the referee needs on the right move (§4 of the ceiling report).
- **Sprint window:** the S3 change (count^4 targets, ×3 on disagreements, policy-only, hit-rate pre-screen) was relayed at 16:10. For S2: down-weight the POLICY loss on fresh 100-sim self-play (≈0.3) and keep its value loss.

## 1. Knobs, ranked by how likely each limits the gain

| # | knob | evidence | recommendation |
|---|---|---|---|
| 1 | policy-target operator = 100-sim visits (`reslib.py:211`, `gpu_sp.py:192`, `sp.py`) | referee-100 choice = deep move 47% vs prior top-1 68% (300 live positions) | short term: low policy weight on 100-sim self-play, policy from ≥800-sim labels (S3). Long term: playout-cap randomisation / Gumbel targets |
| 2 | passes / steps, no held-out split | v6 17 passes → +20 pp; v7 98 passes → 54% vs v6 | ~15-25 passes at batch 2048 (≈2.5-4k steps on 312k); hold out 5% of games; stop on held-out lpol/lval |
| 3 | replication of small sets (teacher ×3/×5, endgame `rep`, oracle ×3, live ×2) | seen 100s of times at 98 passes | cap the effective passes per small set (≤20) via sampling weights, not copies |
| 4 | policy-target sharpness | CE on visits teaches ln(ratio): 40% vs 25% → 0.5 logits; the referee needs ≈ +2 | cnt^(1/T), T ≈ 0.25, for deep labels only (never for noisy 100-sim visits) |
| 5 | value-target blend | 0.7z + 0.3q, z-only at low material, root_q for oracle rows | keep. The oracle rows' root_q shares v2's endgame blind spot, so their weight should stay small (it's 300 rows) |
| 6 | MAGENTA ×2 weight | v6 MAGENTA 55% vs v2 in the quick gauntlet; v7 MAGENTA 44% vs v6 | keep ×2; add a per-colour gate (below) |
| 7 | SWA window (last 20%, every 50 steps, LR floor 0.1×) | within-run averaging measured positive earlier (v20/v30 midpoint) | fine as is; `retrain_fast.py`'s SWA path rebuilds the full tensor set every 50 steps (slow, not wrong) |
| 8 | LR (cosine, max 1e-4 for fine-tunes, OneCycle 6e-4 for full retrains) | no evidence it limits | leave |

## 2. Bug / mismatch audit (all checked; none found)

| item | where | verdict |
|---|---|---|
| D4 symmetry on planes vs policy | `azlib.py:_d4`, `train()`; `sprint2/ft.py step_fn` | consistent: planes gathered with P⁻¹, indices mapped as P[from]·64 + P[to]; `symcheck.py` validated 12,192 checks |
| value sign on the 2-action turn | `sp.py:64` z = +1 iff the position's team won | correct (per-position team, not per-ply alternation) |
| canonical action indices | oracle npz built with `action_index(a, env.current_team)`; S3 uses `mcts.visit_distribution` (canonical) | consistent with the policy head |
| live_opp imitation targets | `ft.py` imitates only where z > 0 | correct: live_opp has z > 0 in 40% of rows and live_ours in 62% (they sum to ≈1), so live_opp z is from the OPPONENT's view and only winning opponents are imitated |
| obs scaling | stored ×8 uint8; low-material counts use /8; oracle rows `rint(obs*8)` | consistent |
| BN recompute after SWA | uniform 8,192 samples, momentum=None | fine |
| shared init | `torch.manual_seed(1)` in every retrain | not a bug; it's why cross-run weight averaging fails while within-run SWA works |
| no held-out split | all trainers | real gap: it hid v7's overfit |

Not verifiable in the time available: the script that built `agents/live/*.pkl` isn't in the repo (only the pickles), so the one-hot move encoding of live_opp was checked only indirectly (the first record decodes to a legal-looking start move, index 3300, CYAN to move).

## 3. Sprint-window actions (16:28)

1. **S3 (relayed 16:10):**
   - targets ∝ cnt⁴;
   - ×3 weight where the 800-sim best ≠ v6 top-1;
   - policy-only loss;
   - before the arena, screen the referee-100 oracle-hit rate on a held-out `lab_*.pkl`, and go on only if it rises by ≥10 points.
2. **S2:** set the policy-loss weight on `fresh` (and old `sp`) rows to about 0.3 and keep their value loss at 1. Fresh v6 outcomes are new value information; their 100-sim visit targets are not new policy information. Keep steps at ≤ ~20 passes.
3. **Never ship on a parent-only head-to-head:** also check the per-colour split. v7's MAGENTA was 44%.
4. **If nothing passes, keep v6** (2234, rank 13, 6/8 live).

## 4. After the contest: pipeline restructure, in priority order

1. **GPU batched self-play engine** (`agents/gpu/gpu_sp.py`): 10-100× more games per hour. Everything below needs it.
2. **ExIt with deep targets:**
   - KataGo playout-cap randomisation (about 25% of moves at ≥800 sims recorded as policy targets, the rest at ~100 sims for value/outcome only) or Gumbel completed-Q targets;
   - sharpen deep targets (T ≈ 0.25);
   - track the referee-100 oracle-hit rate as the main policy metric. Ceiling: v2@800 beats v2@100 at 92.5%.
3. **Exact endgame DP teacher:**
   - finite-horizon backward induction over rounds (the round counter makes the game graph acyclic);
   - K+X vs K first (13.7M states for rounds 9-16), then K+X vs K+Y;
   - needs vectorised rules (numpy/GPU), since Python env calls are too slow.
   - Its values replace search-Q/outcome labels exactly where the oracle is blind.
4. **Evaluation ladder:**
   - (a) a fixed held-out position set with deep labels (oracle hit rate, Q-loss, the lowmat sign suite): 1-minute screens;
   - (b) a per-colour gauntlet (v2, v6, hand_v4f, distill_v5hf, stopgap), reported as CYAN/MAGENTA separately;
   - (c) the gatekeeper arena only for finalists;
   - (d) one shared arena implementation (the in-loop arena read ~10 pp high).
5. **Data hygiene:**
   - hold out games;
   - recency-weighted replay window instead of copies;
   - per-source pass budgets;
   - dedupe repeated openings;
   - log passes per source in every run.
6. **Architecture, last:** widen the residual (32×2 → 48×2) within the ~3 ms budget only after targets and data are fixed. Nothing so far shows a capacity limit.
