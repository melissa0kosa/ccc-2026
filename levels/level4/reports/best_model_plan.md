# Best achievable model under the contract: ranked levers and post-contest plan

Author: f7 · 2026-10-23, written 15:40-16:30 UTC during the contest.
Companion report: `perfect_play_ceiling.md` (session -45) covers the theoretical ceiling, the per-phase gap decomposition and how to exploit the fixed search.
Labels used below: **[obs]** = measured in this repo, **[lit]** = published result (link), **[est]** = my estimate.

## TL;DR
1. **The game is exactly solvable in principle.**
   - (round, side to move, actions_left) strictly advances with every action, so no state can recur. Everything is destroyed at the end of round 26 (`constants.SHRINK_ROUNDS`).
   - So the game is a finite DAG of at most 26 × 2 × 2 = 104 actions, and exact values come from plain backward induction layer by layer. No retrograde fixpoint iteration is needed.
   - The observation encodes the full state: pieces, the locked ring (which phase), `firewall_urgency(round)` (distinct per round within a phase), actions_left and side.
     - The referee search takes exactly (board, team, actions_left, round) (`mcts.search_action`).
     - The env rebuilds the board from observation planes (`Board.from_observation`).
   - If so, an exact value function is representable, and the network capacity is the only limit.
2. **Perfect play for us** means a value head equal to the game-theoretic value and a prior concentrated on optimal actions. With both, the referee's 100-sim PUCT plays optimally.
   - We can get this *exactly* in low-material and late phases by solving them, and *approximately* in the middlegame through deep-search teachers.
3. **Ranking by expected Elo per hour of compute (post-contest):**

   | # | lever | Elo/hour | confidence |
   |---|---|---|---|
   | 0 | Fast engine plus batched MCTS (enabler: multiplies every lever below) | n/a, ×10-50 throughput | high |
   | 1 | **Distil deep search into the move ORDERING (ExIt)**, starting with an **opening book in the weights** for actions 1-8 | **highest**: the measured ceiling is v2@800 vs v2@100 = 92.5% | medium-high (measured ceiling; achieved fraction unknown) |
   | 2 | Exact endgame DP used as a **teacher** (relabel value and policy) plus endgame-start games | high per hour (cheap compute), smaller share of the gap | medium: no published Elo for TB-as-teacher; based on our own endgame-loss data |
   | 3 | Training-only auxiliary heads (free at inference) | medium-low on 8x8 | medium |
   | 4 | Bigger or smarter residual within the caps (global pooling, 6-8 blocks) | medium, needs data from levers 1-2 | medium |
   | 5 | Value or prior temperature, global sharpening, test-time augmentation, warm factors | ≈ 0 (measured) | high (measured) |

4. **The biggest single difference from 2400+ teams is probably data volume, not architecture [est].**
   - Our pure-Python engine gives about 20k games per hour on 5-6 cores.
   - A team with a compiled engine gets 10-100× more self-play and deep-search labels.

## 1. Constraints that shape the answer
- **Contract [obs]** (`docs/submission.md`, `gate/report.py`):
  - one ONNX file, at most 5 MiB;
  - at most 1024 nodes and 512 initializers (gate caps; the server also has unpublished structural budgets);
  - allow-listed ops with constant shapes. Data-dependent Gather/Reshape are banned, but `MatMul`/`Softmax`/`Transpose` are allowed, so attention is possible;
  - inference under limits of about 2 s CPU for the upload check and about 15 s CPU per match (Level 3 figure).
- **Referee [obs]:**
  - PUCT, 100 sims, c_puct 1.5, batch 8, Dirichlet α 0.3 / ε 0.15;
  - opening actions sampled at temperature 1.0 for actions 1-4 and 0.5 for actions 5-8, greedy after that.
- **Live champion az_res4f_v2 [obs]** (`agents/az/RESULTS.md`, `gate/RECOMMEND.md`):
  - Architecture: a fixed hand-feature base (FastNet4, exact threat and firewall features) plus a learned residual: 32-channel conv tower, a bilinear from→to policy (k=16, plus a 64×64 term) and a value MLP over the conv map, global mean and base features.
  - Size and speed: 278 nodes and 321 initializers, 1.27 ms per batch-8 call, 1.28 s CPU per game for both sides.
  - Training: 3000 steps offline on 373k positions (100-sim self-play plus endgame-start games, value target 0.7·z + 0.3·Q), 45-55 s on the GPU.
  - Live rating: 2114 (rank 14) at 15:32. The top team was at 2551 at 15:09.
- **Headroom [obs/est]:**
  - **Initializers are the binding cap** (321/512). A conv with batch norm folded in costs 2 initializers, so roughly 90 more conv layers fit. Nodes are at 278/1024.
  - CPU allows about 2-3× more per call (to about 3 ms per batch-8 call). 64ch × 4 blocks alone measured 4.55 ms, which is too slow; 32-48ch with 6-8 blocks fits.
- **What has already failed against v2 [obs]:**
  - test-time augmentation (2/4/8 symmetries: +0 to 5 pp, and 2-5× the CPU);
  - value and prior temperatures, residual scaling, warm factors (≤ +9 pp, Wilson lower bound < 55%);
  - distillation students (about 50%) and the prior agent (47%).
  - Conclusion: **inference-time tricks are exhausted. Only better training targets or more data move the needle.**

## 2. How close to perfect play can we get?
**Exactly solvable state classes** with Kernel plus at most 1 piece per side (5 piece types). A layer is one value of (round, side, actions_left); each phase has 8 rounds, so 32 layers.

| phase (rounds) | open board | placements per layer | layers | states | feasibility |
|---|---|---|---|---|---|
| 25-26 | 2x2 | < 1k | 8 | trivial | Python |
| 17-24 | 4x4 | 16·15·71·66 ≈ 1.1M | 32 | ≈ 36M | numba/numpy: minutes; pure Python: too slow |
| 9-16 | 6x6 | 36·35·171·166 ≈ 36M | 32 | ≈ 1.1B | numba/C: about 1-3 h; the K+1 vs K subset is only about 7M |
| 1-8 | 8x8 | ≈ 384M | 32 | ≈ 12B | skip (rare at low material) |

**Notes:**
- K vs K is decided immediately by the bare-Kernel rule (rules.md: Chebyshev ≤ 2, then ring, then MAGENTA). K+1 vs K is the core endgame class.
- `agents/hand/eg_override.py` already encodes the bare-Kernel subset exactly. It was verified by brute force: 0 errors on 3000 positions.
- **Measured ceiling for "same net, better ordering"** (from -45, `perfect_play_ceiling.md`, 300 positions from 16 live games):
  - Search scaling with the same model: **v2@800 vs v2@100 = 37/40 (92.5% [80, 97])**, and v2@300 vs v2@100 = 26/40 (65%).
    - So a net whose prior and value reproduce its own 800-sim search would gain roughly +300-450 Elo under the referee.
    - That's the ExIt ceiling for one deep-teacher round [est from the 40-game CI].
  - Q-loss of the referee-100 decision against the v2@800 oracle, 0.79 per game in total:

    | phase | share of the gap | Q-loss per decision | oracle hit rate |
    |---|---|---|---|
    | warmup (actions 1-8) | **43%** | 0.085 | 36% |
    | middlegame | 44% | 0.035 | 51% |
    | low-material | 13% | understated: the oracle shares v2's endgame blind spot | |

  - Boosting the oracle move by +2 logits at the root → 85% hit and -72% Q-loss; +10 → 99% (97% in the warmup) and -95%.
  - **Global** sharpening ×1.5-3 is flat: hit rate 46-47%, Q-loss -12% at best. That matches the failed temperature and warm-factor candidates.
    - The prior's top-1 agrees with the oracle only 68% of the time, so sharpening amplifies the wrong moves as much as the right ones.
    - **Only label quality on the specific move helps.**
  - The prior agent's 2.76k relabelled positions lowered post-warmup agreement (0.659 → 0.615), so expect to need tens of thousands of labels mixed into the full buffer.
- **Middlegame:** not solvable, so we approach it by teacher depth.
  - Session -45 is measuring how much a perfect (oracle) prior would gain over v2 at the referee's 100 sims, by phase. Its 800-sim searches on our live games are being saved as `agents/distill/data/oracle800_az_res4f_v2_<match_id>.npz`, with keys obs, legal, visits, root_q, phase, round and actions_left.
  - Caveat from -45: the low-material labels in that data inherit v2's endgame blind spot, so the DP must override them.
- **Practical ceiling [est]:**
  - exact play in every K+≤1 position from round 9 on;
  - a near-oracle prior and value in K+2/K+3 endgames via a DP-labelled teacher;
  - middlegame strength bounded by the depth of the teacher and data volume, not by the caps.
  - The function-approximation error of a 32-48ch net on an 8x8 board is small next to our current label noise (100-sim visits, outcome z).

## 3. The levers in detail
### Lever 0: fast engine plus batched MCTS (enabler)
- **Today [obs]:** 1.3-2.1 s CPU per self-play game; one 100-sim search takes about 43 ms on the loaded machine.
- **Plan:**
  - Port move generation and the apply/undo step to numba or C.
  - Run N games in lockstep, collecting leaf batches across games for one GPU forward pass.
  - Keep `colony_one/mcts.py` semantics exact, and test parity against the referee on 1000 positions: identical visit counts with seeds fixed.
- **Expected:** 10-50× more positions per hour [est]. Every lever below scales with this.

### Lever 1: distil deep search into the move ordering, opening book first
- **Why the opening first:**
  - It's 43% of the measured gap, with the highest Q-loss per decision (0.085).
  - The referee *samples* actions 1-4 at temperature 1.0 and 5-8 at 0.5, in proportion to visits. So any visit mass on bad moves is played with that probability, and a sharp, *correct* prior is rewarded directly.
  - The position set is small and repeated: one fixed start scene, 8 sampled actions. A 32ch net can memorise it as an opening book in its weights.
- **Recipe:**
  1. Enumerate positions reached in actions 1-8 across self-play and tournament replays, deduplicated by board hash (`gate/garena.py` already computes `h8`/`hb` hashes).
  2. Label each with a deep search: 1600-6400 sims, or alpha-beta from `agents/distill/teacher.py` cross-checked against MCTS.
  3. Policy target = one-hot or very sharp on the deep move, with a margin of ≥ 2 logits over the next move (the measured +2 → 85% hit). Value target = deep root Q.
  4. Upweight these positions in the training mix, but keep the full buffer so other phases aren't forgotten (the prior agent's 2.76k-label regression).
- **Then extend to the middlegame:** the ExIt loop with playout-cap randomisation (lever 1b recipe below).
- **Quick check before investing:** gate a model trained this way on only the opening against v2. The warmup accounts for 4 of the 19 decisions per game but 43% of the Q-loss.

### Lever 2: exact endgame DP as a teacher
- **Why a teacher and not only a lookup:**
  - Under the caps, a gated in-graph lookup can only cover tiny rule-expressible classes (what `eg_override` does).
  - Using the DP to *relabel* training data teaches the net the endgame itself, and that generalises to K+2/K+3 positions and to the transition into the endgame. Firewall escape decided 8 of our first 18 live games.
- **Steps:**
  1. Solve the 2x2 and 4x4 layers, then 6x6, for K+≤1 vs K+≤1.
  2. Store value (win/loss) plus distance-to-end and the set of optimal actions.
  3. For every buffered position in a solved class, set the value target to ±1 exact and the policy target to uniform over optimal actions (or DP-scored softmax).
  4. Generate endgame-start games (`agents/az/sp_end.py` already exists) and label them by the DP.
  5. Optionally use the DP to rescore game outcomes z for the moves leading into solved positions.
- **Evidence [obs]:** v2's low-material sign accuracy is only 0.79 (MSE 0.487) on 299 live positions (`gate/RECOMMEND.md`), and adding outcome-labelled endgame data already moved az_res4f v1 → v2 (endgame MSE 0.39-0.41 → 0.32-0.34 on 3840 held-out endgame positions). Against az_res_v20 that went from 50% → 76% head-to-head, but v2 also got more self-play data, so not all of the +26 pp is from the endgame (`agents/az/RESULTS.md`).
- **Literature:** see [§6](#6-literature-summary).

### Lever 1b: expert iteration with deep targets (middlegame)
- **Recipe [est, KataGo-style]:**
  - Per generation, self-play at 100 sims with temperature, but on about 25% of moves run a full 800-1600-sim search and record **only** those as policy targets (playout-cap randomisation).
  - Value target = 0.7·z + 0.3·Q_deep, overridden by the DP in solved classes.
  - Train 3000-6000 steps, gate with SPRT against the best, repeat.
  - Seed generation 1 with -45's oracle800 file.
- **Throughput:**
  - Without lever 0: 800-sim labels cost about 0.35 s each per core [obs-derived], about 30k labels per hour on 3 cores, so 1 generation per 1-2 h.
  - With lever 0: several generations per hour.
- **Expected:** +30-70 Elo per round early on, shrinking later [est]; literature in §6. This was our sprint question, and in the 25 minutes we had it was not feasible.

### Lever 3: training-only auxiliary heads
These heads are **not exported**, so they cost nothing at inference:
- moves-left / rounds-to-end;
- "own Kernel destroyed by the firewall" in the next shrink;
- per-square occupancy at game end, an ownership analogue;
- the opponent's next action, as in KataGo.

They regularise the shared trunk toward exactly the concepts our losses come from (firewall timing, Kernel safety).

### Lever 4: architecture within the caps
- 32ch × 6-8 blocks, or 48ch × 4-6. Check ≤ 3 ms per batch-8 call and the validator under load.
- Add global-pooling bias layers (KataGo) in the residual: cheap (Reduce + Gemm + Add) and good for global facts such as round, ring and material balance.
- Keep the fixed hand base. It gives exact tactical features for free, and step-0 parity is 0.0.
- Attention fits the op allow-list but is node-hungry; low priority.
- Bigger nets only pay off with the data from levers 0-2.

### Lever 5: inference-time tweaks
Measured as exhausted (§1). Drop them.

## 4. What 2400+ teams likely do differently [est, unverifiable: other teams' replays are not exposed]
1. **More self-play and deeper labels through a faster engine.** This is the most likely gap.
2. **Proper AZ from scratch with many generations.** Our nets are residuals on hand features, trained mostly offline over a few generations.
3. **Exact handling of endgame and firewall timing.** Our replays show losses concentrated there.
4. **Larger nets near the caps**, trained on that data.

## 5. Post-contest plan (the game stays playable)
| step | work | compute | gate / success criterion |
|---|---|---|---|
| P0 | Fixed anchor ladder: stopgap_v1, hand_v2, az_res_v20, az_res4f_v2, 400 games each, for absolute Elo | 1 h CPU | stable ladder ±25 Elo |
| P1 | Lever 0: numba/C move generator plus batched lockstep MCTS; parity test against `colony_one/mcts.py` | 3-6 h dev | identical visits on 1000 seeded positions; ≥ 10× games/h |
| P2 | Endgame DP, K+≤1 vs K+≤1 (2x2 → 4x4 → 6x6); brute-force check like `agents/hand/eg_verify.py` | 1-3 h CPU after the port | 0 errors on 10k random positions |
| P2b | **Opening book in the weights** (lever 1): dedupe actions 1-8 positions from self-play and replays, label at 1600-6400 sims, train with a margin of ≥ 2 logits, upweighted, on the full buffer | 1-2 h CPU without P1, minutes with P1 | warmup oracle hit ≥ 85% on held-out games; ≥ 60% against az_res4f_v2 |
| P3 | Relabel the buffer (373k + oracle800) with DP values and policies in solved classes; endgame-start games; retrain v2's recipe → v3-DP | 30 min | ≥ 60% against az_res4f_v2 (SPRT) |
| P4 | ExIt loop: 20k games per generation at 100 sims, 25% of moves at 800 sims, DP overrides, auxiliary heads; train, then SPRT gate | about 1 generation per hour | +30 Elo per generation on the ladder until it plateaus |
| P5 | Architecture sweep under the caps (32×6, 48×6, plus global pooling) on the P4 data | 2-4 h GPU | ≥ +40 Elo on the ladder at ≤ 3 ms per batch-8 call |
| P6 | Extend the DP to K+2 vs K+≤1 on 4x4 and 6x6 if P3/P4 show endgame losses remain | hours (numba/C) | lowmat sign accuracy ≥ 0.98 |

## 6. Literature summary
Most of these numbers come from **19x19 Go** (KataGo) or Hex (ExIt). On a small board (Connect 4, Clausen et al. 2021) most add-ons gave "mostly only small" gains, so read them as an upper-bound ordering, not as transferable Elo.

| lever | published effect | source | relevance to us |
|---|---|---|---|
| 1. Tablebase as teacher | Lc0 TB rescoring rewrites the value target from the first TB position. **No published Elo.** The TCEC 15 winner used 7-piece rescoring, confounded with an LR change. | [Lc0 2018](https://lczero.org/blog/2018/09/tb-rescoring/), [Lc0 2019](https://lczero.org/blog/2019/06/whats-going-on-with-training/) | Thin external evidence; our case rests on our own endgame-loss data (§3, lever 1). |
| 2. ExIt / deeper teacher | Visit distribution vs chosen move: **+50 ± 13 Elo**. 3 more DAgger batches of 100k moves: **+120 Elo**. 2× teacher iterations: 56% (about +42). | [Anthony et al. 2017](https://ar5iv.labs.arxiv.org/html/1705.08439) | Direct analogue. Supports +30-70 per round early on. |
| 2. Playout-cap randomisation | 1.37×, **-87 Elo** without it (p 0.25, 600/100 → 1000/200 playouts) | [KataGo, Wu 2019](https://arxiv.org/abs/1902.10565) | Matches our "100-sim play, deep targets" design. |
| 2. Forced playouts + policy-target pruning | 1.25×, -53 Elo without them | KataGo | Optional. |
| 2. Gumbel (training side only) | Gumbel MuZero learns at 2 sims, while plain MuZero fails at ≤ 16. MiniZero: at 200 sims plain AZ ≥ Gumbel; on 8x8 Othello at equal time, Gumbel at 2 sims ≈ AZ at 200. | [Danihelka 2022](https://iclr.cc/virtual/2022/poster/6418), [MiniZero](https://arxiv.org/abs/2310.11305) | Only if label compute is the bottleneck. The referee's PUCT is fixed. |
| 3. Auxiliary targets | Ownership + score: 1.65×, **-190 Elo** without (largest KataGo factor). Opponent-move head: 1.30×, -74. Connect 4: small, other auxiliary targets didn't help. Lc0 moves-left: no measured Elo. | KataGo; [Clausen 2021](https://www.scitepress.org/Papers/2021/102459/pdf/index.html) | Cheap to try (training-only). Expect small gains on 8x8. |
| 4. Global pooling | 1.60×, **-176 Elo** without | KataGo | Probably smaller on 8x8 (the receptive field covers the board fast), but our residual already uses a global mean in the value head, so add it to the policy trunk too. |
| 4. Game-specific input features | 1.55×, -161 Elo without (bundled). Chess: new inputs plus value-loss changes up to +180 over AZ. | KataGo; [Czech 2023](https://arxiv.org/abs/2304.14918) | We already have this: the hand-feature base. |
| 4. SE / recurrent / attention | SE: informal "significantly stronger" (Lc0 forum), mixed on Connect 4. Recurrent: 92% vs 84% on chess puzzles (not play). Lc0 BT4 transformer: +270 raw-policy Elo, but at ~190M params. | [Schwarzschild 2021](https://arxiv.org/abs/2106.04537), [Lc0 2024](https://lczero.org/blog/2024/02/transformer-progress/) | Low priority under 1024 nodes and 512 initializers. |
| 5. Value target z vs Q | Soft-Z / A0C / greedy backups "train faster and perform better" than z (abstract only, no numbers). **No study under a fixed-sim referee.** | [Willemsen 2021](https://link.springer.com/content/pdf/10.1007/s00521-021-05928-5.pdf) | Our own test: 0.5z+0.5Q gave no gain (v31). |

## 7. Open questions
- The server's structural budgets beyond 1024/512, and its CPU per match in Level 4, are not published.
- The low-material share (13%) is a lower bound, because the 800-sim oracle shares v2's endgame value. Re-measure it against the DP once P2 exists.
- How much of the 92.5% search-scaling ceiling distillation actually captures. The 40-game CI is wide ([80, 97]), and the gate ladder (P0) should track the fraction achieved per round.
