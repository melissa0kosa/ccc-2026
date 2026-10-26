# Search to training data (NNUE, AlphaZero variants) and opening play under forced sampling, mapped to az_res / distill / prior / hand

Written 2026-10-23, about 14:55 UTC. Internal sources are cited by file path; web sources by URL.
Local evidence comes from `levels/level4/reviews/hand_v2.md`, `levels/level4/agents/*/RESULTS.md`, `levels/level4/DEPLOYS.md`, the referee `data/colony-one-env/colony_one/mcts.py` (read only) and the distill teacher `agents/distill/teacher.py` (read only).
I also ran one small toy simulation of the referee root search (1 core, under 1 s), described in Q3:
`C:/Users/melis/AppData/Local/Temp/claude/C--Users-melis-Downloads-cloudflight-coding-contest/2537b5b1-8fcc-4ac8-8fd3-b627f7b7b210/scratchpad/puct_toy.py`.

## Q0. What should the team do in the remaining ~1-1.5 h, ranked by expected gain per hour?

### Takeaway
1. Ship warmup-only prior sharpening inside whichever model is uploaded next. It is about 10 minutes of work and has measured local evidence.
2. The biggest single measured edge is distill_v3hb, at 78% vs hand_v3. Its latency, not its strength, is the blocker.
3. A tiny "opening book" covering the T = 1.0 window (actions 1-4) is the only fix for the no-signal opening blunders (6509), at about 45-60 min.
4. AZ-side changes such as playout-cap randomization or rebasing on hand_v3 do not fit in the remaining time.

### Cited Findings
- Warmup cost: 30 of 52 warmup decisions were not the greedy-100 move. The mean MCTS-800 Q-loss per warmup decision is 0.079, against 0.035 after the warmup. Warmup sampling caused loss 6177 and contributed to loss 6509. — `reviews/hand_v2.md` §4 W1
- Warmup-only sharpening ×1.5 at 6177 a7 under the referee config (40 seeds) raised P(best move) from 0.78 to 0.97, and ×2 to 1.00. It does not help at 6509 a2/a6, where the prior has no signal. — `reviews/hand_v2.md` W1
- Global prior ×1.5 (hand v4b vs v4a) gave 103/200 vs 104/200: no post-warmup harm. The hand_v4 bundle already contains "opening sharpening x(1+0.5*warm)". hand_v4f scored 109/200 = 55% vs hand_v3. — `agents/hand/RESULTS.md`
- distill_v3hb (a residual on hand_v3 FastNet) scored 47/60 = 78% [66,87] vs hand_v3. But it costs 6.4 ms/call at b8 under load (hand_v3: 1.3 ms) and passed the validator on only 1 of 3 tries ("too slow to check", 3 s wall-clock budget). — `agents/distill/RESULTS.md`
- az_res_v6 scored 141/200 = 70% vs hand_v2. hand_v3 scored 83/120 = 69% vs hand_v2, so the two are about equal against the same reference. v7 and v8 (63%, 70% over 60 games) are inside the ±12-15-point noise of 60-game arenas. — `agents/az/RESULTS.md`, `agents/hand/RESULTS.md`
- Upload cost: about 2 swap forfeits per upload. Only edges clearly above +50 Elo should go up. +20 pp edges are allowed until 16:20, and there is a hard stop at 16:40. — `DEPLOYS.md`
- The prior pipeline has no arena results yet ("filled in below as runs finish"). — `agents/prior/RESULTS.md`

### Inferences
Ranked list. "Gain" means expected live win-rate change, "cost" means team time.

1. **Warmup-only sharpening, ×2 in actions 1-4 and ×1.5 in actions 5-8.**
   - Cost: 10 min plus an offline "warmup Q-loss" check (below). No standalone upload: bundle it with the next model.
   - Gain: in the toy model (Q3), with a prior signal, P(best) at T = 1 rises from about 0.35 to 0.55-0.75, and expected Q-loss roughly halves.
   - Upper bound for the whole warmup fix: the excess Q-loss is 0.044 per decision × about 4 own warmup decisions = 0.18 Q per game, roughly 9 pp of win probability with p = (1+Q)/2. Realistically we recover a third to a half of that.
   - The exact window detector already exists in `hand_v2.md` W1. Add the `actions_left` plane (plane 13 in `encoding.py`) if a per-block split is wanted. The round number also distinguishes rounds 1 and 2: the block boundary is action 4 = end of round 1.
2. **Make distill_v3hb deployable, then stack the warmup sharpening on it.**
   - Gain: largest measured edge (+28 pp head-to-head vs hand_v3). Cost: 20-40 min.
   - Options: shrink the trunk from 32×4 to 32×2, or use 24 channels. Re-measure the dry run against the 3 s validator budget.
   - Gate it through `DEPLOYS.md` against the current seat.
3. **Opening book for the T = 1.0 window** (Q3). Cost: 45-60 min. It is the only way to fix 6509-type positions, where the prior has no signal. Low risk: it only fires on exact-match positions.
4. **Offline warmup metric instead of arenas, for any opening change.** Cost: 15 min. It turns ±12-15-pp arena noise into a direct, low-variance measurement.
   - Take a fixed set of warmup positions: the 52 live ones plus about 300 from self-play with the referee config.
   - For each candidate model, run referee-config MCTS-100 for about 10 seeds per position.
   - Report the expected Q-loss against one frozen reference: alpha-beta teacher scores, or MCTS-800 with the strongest model.
   - Cost per model: about 350 × 10 × 35 ms ≈ 2 min on 1 core.
5. **Calibrate the distill value scale** (Q1). About 10 min, modest gain, and only worth it if more distill training happens anyway.
6. **Out of scope for today:**
   - az_res rebased on hand_v3 with teacher-value targets and playout-cap randomization: it needs several generations, more than 1.5 h.
   - Ensembling separate bases in one graph: the initializer and latency budget does not allow it (Q4).

### Gaps
- No live evidence yet that warmup sharpening changes live results. The only measured effect is the single-position test at 6177 a7 and the toy model.
- Converting Q-loss to win-rate with p = (1+Q)/2 is a heuristic, and the MCTS-800 reference shares hand_v2's blind spots (`hand_v2.md` §5 caveat).

## Q1. NNUE-style data generation (Stockfish nnue-pytorch, YaneuraOu gensfen, Fairy-Stockfish): what applies to the distill pipeline?

### Takeaway
NNUE training targets a blend of the search score (mapped to a win-rate by a scaled sigmoid) and the game result. Our distill value label, 0.75·tanh(best/3) + 0.25·result, is already the same recipe (λ = 0.75). The cheap transferable piece is calibrating the score scale. NNUE's "quiet positions only" filtering should not be copied: our net is called at every MCTS leaf, with no quiescence search behind it.

### Cited Findings
- nnue-pytorch loss, verbatim from the training module:
  - The network score is converted to a win-rate with `q = (scorenet - in_offset)/in_scaling; qf = 0.5*(1 + sigmoid(q) - sigmoid(qm))`. The search score is converted the same way with `out_offset`/`out_scaling`.
  - The target is `pt = pf*actual_lambda + t*(1 - actual_lambda)`, with `t` = game outcome.
  - Lambda is scheduled linearly over epochs, from `start_lambda` to `end_lambda`.
  - The loss is `|pt - qf|^pow_exp`, with an optional asymmetry term `qp_asymmetry`.
  - Signature defaults include `max_epoch=800`, `gamma=0.992`, `lr=8.75e-4`, and 100M positions per epoch at batch 16384.
  - Source: [nnue-pytorch lightning_module.py (mirror copy)](https://isip.piconepress.com/projects/ai_chess/resources/pro_player_models/nnue-pytorch/model/lightning_module.py)
- Lambda semantics, as documented in nnue-pytorch: 0 means pure game results, 1 means pure search scores, and values in between interpolate. Early nnue-pytorch runs all used λ = 1.0, i.e. trained to the evaluation and not to the game result. A later post notes a bug in the recorded game outcomes that made λ ≠ 1 runs suspect. — [Talkchess "Pytorch NNUE training"](https://talkchess.com/viewtopic.php?p=872983), [follow-up post](https://talkchess.com/viewtopic.php?p=872755)
- The official nnue-pytorch "Basic training procedure" page lists only example sizes: epoch-size 1,000,000, validation 100,000, batch 16384, 2 epochs. It calls this "a development example, not a recipe for a competitive network." — [nnue-pytorch wiki: Basic training procedure](https://github.com/official-stockfish/nnue-pytorch/wiki/Basic-training-procedure-(train.py))
- Team distill facts:
  - The value label is 0.75·tanh(best/3) + 0.25·result.
  - The policy label is softmax(scores/T). T = 0.3, 0.5 and 1.0 were about equal.
  - More data and "child" value samples (positions after up to 8 random root actions, labelled with the teacher score) are what helped: v2 with child samples beat v2n without them 38/60.
  - The teacher itself scores 19/20 vs stopgap_v1, so it has a lot of headroom over the student.
  - Source: `agents/distill/RESULTS.md`
- Teacher root scoring:
  - Every legal root action is searched with alpha = best − 3.0. Moves more than 3 below the best get only an upper bound.
  - Depth = actions_left + 2 (the rest of our turn plus the full opponent turn), then a captures-only qsearch.
  - Source: `agents/distill/teacher.py` lines 230-245

### Inferences
- **λ mapping.** Our label is the NNUE recipe with λ = 0.75 and a tanh link. Since tanh(x/3) = 2·sigmoid(2x/3) − 1, it is a sigmoid scale of 1.5 in teacher-score units.
  - The NNUE practice is that the scale must be calibrated to real outcomes.
  - 10-minute test: on the teacher-behaviour games, fit `s` in `result ≈ tanh(best/s)` (1-parameter Texel/BCE fit) and compare with s = 3.
  - If s is off by more than 1.5×, the 0.75 eval term is systematically over- or under-confident. Relabel and retrain; labels are cheap to recompute from the stored scores, assuming `best` is stored.
- **λ itself.** NNUE's history supports a high λ (pure eval worked). The team's own AZ offline experiment supports Q-mixing over outcome-only (Q2).
  - Keep λ in 0.75-1.0. A λ sweep is not worth the arena time today.
- **The upper bounds at the root are harmless.** With T = 0.4, a move 3.0 below the best gets weight e^(−7.5) ≈ 5e-4, so the bounded scores never matter in the policy label. No fix needed.
- **Do NOT apply NNUE's quiet-position filter.**
  - In Stockfish, the NNUE eval is consumed at quiescence leaves of a deep alpha-beta search, so tactical positions are resolved by search before the eval is called. This is my own reasoning; I could not fetch the source this session.
  - Our net is the leaf evaluator of a 100-sim MCTS with no qsearch. The live losses come from exactly the tactical patterns a static eval misses: the 2-action strikes W2 and W3 (`hand_v2.md`).
  - The teacher's qsearch-resolved labels on tactical positions are the most valuable part of the data. If anything, upweight positions where the teacher's best score differs from the static eval by a large margin.
- **Random-move randomization.** The "child samples" are the team's equivalent of gensfen-style random moves, and they helped.
  - Opening twist: the referee itself randomizes actions 1-8. So the distill data should be generated with the real referee warmup config (T 1.0/0.5 sampling plus Dirichlet), with the current best student as the behaviour policy (DAgger, as in r4).
  - That way the warmup positions the referee actually induces are labelled by the teacher.
- **Data volume.** NNUE scale (100M+ positions per epoch) is irrelevant here. The team's own curve (1.8k → 3k → 6.7k → 11.4k positions, each step a gain) says data is still the binding constraint.
  - Keep a 1-core teacher generator running in the background until about 16:00 if a distill retrain is planned.

### Gaps
- I could not retrieve:
  - the nnue-pytorch default values (`in_scaling`, `out_scaling`, `start_lambda`, `end_lambda`, `pow_exp`), which live in a config file not served by the mirror;
  - the official "Training datasets" page;
  - the YaneuraOu `gensfen` and Fairy-Stockfish `generate_training_data` parameters: depth, random_move_count, random_multi_pv, eval_limit, write_minply. The Fairy-Stockfish wiki page URL returned an empty page.
- My recollection, not source-checked this session: nnue-pytorch's "smart fen skipping" drops positions where the best move is a capture or the side to move is in check. Treat it as unverified.
- No source found on how much data small-variant NNUE nets need.

## Q2. AlphaZero-for-variants practice (CrazyAra/MultiAra, KataGo, Lc0; Polygames, Ludii): warm start, z vs Q targets, plateaus, small-compute tricks. What applies to az_res?

### Takeaway
Published practice is "supervised warm start, then RL", with several cheap self-play efficiency tricks. The only one with a published, quantified gain at small compute that fits our harness is KataGo's playout-cap randomization: 25% of moves get a full search and only those are recorded, worth about 1.37×. The az_res "plateau" (v6-v8) is statistically unresolved. The most actionable az_res change today is a value target from the alpha-beta teacher or deeper search, which the team's own offline experiment already backs.

### Cited Findings
- **KataGo, playout-cap randomization:**
  - A full search is run on p = 0.25 of turns, with N = 600 playouts (later 1000) vs fast n = 100 (later 200).
  - Only full-search turns are recorded as training data. Fast searches disable Dirichlet noise and other exploration.
  - Gain: about 1.37× in training efficiency. Fixed caps N ∈ {100…600} all performed worse.
  - Source: [Wu 2019, Accelerating Self-Play Learning in Go](https://ar5iv.arxiv.org/html/1902.10565)
- **KataGo, forced playouts and policy target pruning:**
  - Each root child gets a minimum n_forced = sqrt(k·P(c)·ΣN), with k = 2.
  - Those forced visits are removed from the policy target afterwards, "since most forced moves are bad".
  - Gain: about 1.25×.
  - Source: [Wu 2019](https://ar5iv.arxiv.org/html/1902.10565); [arXiv PDF](https://arxiv.org/pdf/1902.10565)
- **KataGo, other levers:**
  - An auxiliary opponent-reply policy (weight 0.15) is worth about 1.30×.
  - All the ablation factors multiply to about 9.1×, against a claimed overall 50× reduction vs ELF on the paper's own crude metric.
  - Source: [Wu 2019](https://ar5iv.arxiv.org/html/1902.10565)
- **CrazyAra configuration:**
  - `q_value_ratio` "defines the ratio for mixing the value return with the corresponding q-value"; 0 disables it. The RL config sets `tc.q_value_ratio = 0  # previously 0.15`.
  - The RL config also uses value and policy loss factors of 0.5/0.5 (the supervised default is val 0.01 / policy 0.99, "to prevent overfitting"), non-sparse policy targets, and 1 epoch per RL iteration.
  - Source: [CrazyAra train_config.py](https://raw.githubusercontent.com/QueensGambit/CrazyAra/master/DeepCrazyhouse/configs/train_config.py)
- **CrazyAra history:** CrazyAra was first trained purely supervised (on human crazyhouse games), and an extension continued training with RL on generated engine games. A search summary reports one million self-play games over 18 days on 3 V100s, starting from the supervised net. I saw that only in a search snippet, not in the full text. — [Czech et al. 2020, Frontiers in AI](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2020.00024/pdf); [TUprints record](https://tuprints.ulb.tu-darmstadt.de/13370/)
- **Team AZ facts:**
  - The outcome-only from-scratch AZ reached only 28% vs stopgap.
  - The same net trained offline on 1000 clean teacher games with value = 0.5·z + 0.5·root-Q reached 70% vs stopgap: "Q-mixing + clean teacher data >> outcome-only".
  - az_res uses 0.7z + 0.3Q, ExIt self-play (400 self-play + 100 vs hand_v2 games per generation, 100 sims), and D4 augmentation.
  - Two identical-model sanity arenas gave 14/40 and 18/40.
  - Source: `agents/az/RESULTS.md`

### Inferences
- **The plateau is not proven.**
  - v6 (70% over 200 games) is the only tightly measured point. v7 and v8 at 63-70% over 60 games lie inside the same CI.
  - The structural ceiling is that az_res sits on hand_v2, while hand_v3 alone already matches v6 against hand_v2.
  - Before more generations, play v6 vs hand_v3 directly (60+ games). If it is about 50%, the residual has mostly learned what hand_v3's Texel refit learned, and further az_res work on the v2 base has low marginal value.
- **Value target (combining with distill).**
  - Raise the Q share or replace z with the alpha-beta teacher value on az_res positions. A suggested target is 0.5·z + 0.5·teacher-or-800-sim-Q, matching the team's best offline result.
  - Rationale: with about 500 games per generation, z is a very noisy label for about 35 positions per game.
  - CrazyAra's move back to q_value_ratio 0 happened at million-game scale, where z noise averages out. That does not hold for us. There is a mild conflict here between large-scale practice and our small-data evidence; the team's own A/B is the deciding evidence.
- **Playout-cap randomization fits our harness exactly.**
  - Play 75% of moves at 100 sims, unrecorded, which is cheap and matches the referee. Search and record the other 25% at 400-800 sims, noise-free.
  - This upgrades the policy targets to deeper search at roughly 2× the CPU of 100-sim self-play, instead of 8×. The prior pipeline's premise (800-sim targets are better) is the same idea.
- **Policy target pruning.**
  - Our raw 800-sim visit distributions contain exploration mass that PUCT forces onto bad moves.
  - In the prior pipeline, and in az_res if it continues, a pruned or sharpened target (drop children with Q < Q_best − 0.15 that hold ≤ 2% of visits, or use visits^(1/0.5)) is a 5-minute change. It produces a sharper student prior, which is exactly what the warmup needs (Q3).
- **Polygames and Ludii add nothing actionable for today**, beyond the same "warm start, then RL" pattern; I did not fetch them (see Gaps).

### Gaps
- I did not fetch the Polygames or Ludii-AZ papers, or MultiAra-specific docs, so there are no claims about their value-target mixing or plateau handling.
- I could not retrieve the Oracle "Lessons from AlphaZero part 4: improving the training target" post (HTTP 403). It reportedly compares z, q and their average, but its numbers are not cited here.
- No primary source found on CrazyAra's measured gain from q_value_ratio = 0.15 vs 0.

## Q3. Opening play under forced sampling: what makes the visits concentrate on good moves? Is a sharper prior, deep-search opening distillation, a baked-in book, or value shaping the right fix?

### Takeaway
Sampling visits^(1/T) means that, given a correct top-1, prior sharpness buys most of the gain at T = 0.5 (actions 5-8). At T = 1.0 (actions 1-4) sharpness caps at about 0.75-0.88 P(best), because of the fixed 15% Dirichlet share and the absolute FPU of 0. Where the prior has no signal, the only fixes are better opening labels: deep-search distillation or an exact-match book. Our T = 1.0 decisions are few and mostly enumerable: CYAN a1-a2 from a fixed start, and MAG a3-a4 after the opponents' first turn. So a tiny book baked into the policy is feasible and legitimate. Value shaping via an offset cannot be made consistent and should be skipped.

### Cited Findings
- **Referee mechanics** (`colony_one/mcts.py`, read only):
  - The root action is sampled ∝ visits^(1/T): T = 1.0 for actions 1-4, T = 0.5 for actions 5-8, greedy afterwards. Root Dirichlet(0.3) noise runs over the same window.
  - The noise "is mixed into the priors *after* they are normalised, so its share is fixed at dirichlet_eps no matter how extreme the logits were". The temperature samples visit counts, "which the simulation budget bounds. A model cannot scale its way out of the opening randomisation by shouting."
  - PUCT uses `q = W/n if n > 0 else 0.0`, i.e. an absolute FPU of 0, with c_puct 1.5, and open descents count as losing visits (virtual loss 1).
  - The value perspective flips only at a real turn change. Two actions are taken per team per turn.
- **Lc0 contrast:** the default FPU strategy is `reduction` (relative to the parent eval). The `absolute` strategy's `fpu-value` default is −1.00: "value of unvisited node is assumed to be equal to this value, and does not depend on parent eval." So Lc0's absolute default treats unvisited moves as lost, while the referee's FPU 0 treats them as even. — [Lc0 options wiki](https://github.com/LeelaChessZero/lc0/wiki/Lc0-options) (last edited Feb 2019, may be outdated)
- **Lc0 temperature controls:** Lc0 exposes temperature, tempdecay-moves, temp-cutoff-move and `temp-visit-offset` ("Reduces visits by this value when picking a move with a temperature"), and a policy softmax temperature default of 2.20 ("Higher values make priors … closer to each other, widening the search"). — [Lc0 options wiki](https://github.com/LeelaChessZero/lc0/wiki/Lc0-options)
- **KataGo opening diversity** in self-play comes from:
  - r random opening moves sampled from the raw policy, with mean 0.04·b² (about 14 on 19×19);
  - a move temperature on pruned visits decaying from 0.8 to 0.2 with a half-life of b turns;
  - root noise 0.75·P + 0.25·Dirichlet;
  - branching to alternatives in 2.5% of positions.
  These are all training-side mechanisms, not a book. — [Wu 2019](https://ar5iv.arxiv.org/html/1902.10565)
- **Team measurements:**
  - The prior is flat: median warmup top-1 is 0.06, and the median visit share of the greedy move is 0.17.
  - The 6509 a2 blunder (DE C1→C4) left EX E4 to the 2-action strike H6→H7, H7xE4. That strike is invisible to hand_v2's value and prior (W2).
  - Source: `reviews/hand_v2.md`
- **Teacher horizon:** at an actions_left = 1 root, depth is 3 (our action plus the full opponent turn), plus qsearch. So the teacher sees exactly the 6509 a2/a3-a4 strike pattern. — `agents/distill/teacher.py` line 234

**Toy simulation (my computation).** Script: `puct_toy.py` in the scratchpad (path at the top).
- Setup:
  - Root-only PUCT copying the referee formula: FPU 0, c = 1.5, 100 sims, batch 8 with virtual loss 1, Dirichlet(0.3) at eps 0.15 after normalisation.
  - 25 legal moves; each visit returns the move's fixed Q, with no tree below.
  - Non-best moves are 0-0.4 below the best in Q.
  - Prior = softmax(mult·logits), where the best move's logit exceeds the max of 24 N(0, 0.5) logits by "gap".
- Results, averaged over 300 trials:

| prior signal (gap) | mult | Q_best | P(best) T = 1 | P(best) T = 0.5 | E[Q-loss] T = 1 | E[Q-loss] T = 0.5 |
|---|---|---|---|---|---|---|
| none (0.0) | 1 → 3 | +0.05 | 0.15 → 0.28 | 0.29 → 0.55 | 0.104 → 0.098 | 0.053 → 0.044 |
| some (1.5) | 1 | +0.05 / −0.15 | 0.33 / 0.37 | 0.74 / 0.81 | 0.082 / 0.079 | 0.017 / 0.013 |
| some (1.5) | 1.5 | +0.05 / −0.15 | 0.55 / 0.58 | 0.94 / 0.96 | 0.061 / 0.062 | 0.004 / 0.003 |
| some (1.5) | 2 | +0.05 / −0.15 | 0.74 / 0.69 | 0.98 / 0.99 | 0.039 / 0.052 | 0.001 |
| some (1.5) | 3 | +0.05 / −0.15 | 0.87 / 0.74 | 1.00 / 0.99 | 0.020 / 0.049 | 0.000 |
| strong (3.0) | 2 | +0.05 / −0.15 | 0.88 / 0.74 | 1.00 / 0.99 | 0.019 / 0.049 | 0.000 |

### Inferences
- **The two warmup blocks behave very differently.**
  - At T = 0.5 (a5-a8), any prior with a correct top-1 and ×1.5-2 sharpening makes the referee pick the best move about 95-99% of the time.
  - At T = 1.0 (a1-a4), even a very sharp, correct prior tops out at P(best) of about 0.88 when Q_best > 0, and about 0.74 when Q_best < 0.
  - The Q_best < 0 cap is the FPU-0 effect. When the side to move's best Q is negative, every unvisited move (scored q = 0 + U) beats the visited best, so the search spends 1-2 visits on each of many moves. At T = 1 each of those is about 1-2% sampling mass.
  - The exact numbers depend on the toy's Q spread. The qualitative ordering (T = 0.5 easy, T = 1 capped, FPU penalty when behind) follows from the formula.
- **Sharper prior (team's ×1.5).** Correct direction, cheap, already in hand_v4.
  - Use ×2 in the T = 1 block (actions 1-4) and ×1.5 in actions 5-8. ×3 adds little over ×2 and risks over-trusting a wrong top-1 in a window where the search cannot correct it.
  - The window can be keyed on round: round 1 = actions 1-4, round 2 = actions 5-8, with no locked ring in either.
  - It is useless where the top-1 is wrong (6509), as the "none" rows show.
- **Deep-search distillation on opening positions (prior pipeline).**
  - The generalizing version of the fix. The prior agent already samples warmup positions with the referee config and weights move ≤ 8 by ×2.
  - Changes that help within the hour:
    1. Weight warmup positions ×4-5.
    2. Use a sharpened target, visits^(1/0.5) or KataGo-style pruning, for those positions. The student should be sharper than the 800-sim visit distribution there, because the 100-sim referee search flattens it again.
    3. Label the warmup positions with the alpha-beta teacher, or at least check them against it. hand_v3's 800-sim search inherits hand's 2-action blind spot (W2), while the teacher's depth covers it.
- **Opening book baked into the policy.** Feasible, legitimate and targeted.
  - **Coverage of the T = 1 window:**
    - As CYAN, a1 is the single fixed start position, and a2 follows from our own a1 (a handful of positions once a1 is concentrated). The CYAN T = 1 decisions are fully enumerable with about 5-20 entries.
    - As MAG, a3-a4 depend on the opponent's first turn. Take the most frequent opponent a1-a2 lines from our live replays (76 protocol games across 3 match files, `hand_v2.md` §3) plus our own self-play, and cover the top ~20-40 positions.
    - a5-a8 explode combinatorially; leave them to the sharpened general prior, which works well at T = 0.5.
  - **Labels:**
    - Alpha-beta teacher scores (exact within margin, depth covers 2-action strikes) cross-checked with MCTS-3200 of the best net.
    - If two moves are within about 0.05 of each other, give both the bonus. Diversity does not matter, because the referee already forces it; the bonus only has to concentrate visits on non-losing moves.
  - **ONNX implementation** (exact match, about 3 initializers):
    - Flatten obs to [b, 1024] and compute d = ‖x‖² − 2·x·Bᵀ + ‖B‖² with one MatMul against B [K, 1024].
    - m = relu(1 − d): 1 on an exact match, 0 otherwise, since any differing square or scalar plane gives d ≥ 1. Re-check this on the real encoding: scalar planes are fractions broadcast over 64 squares, so their differences sum to at least 64 × Δ².
    - Add `policy += 4.0 · (m @ Bonus[K, 4096])`.
    - For K = 40: 41k + 164k floats ≈ 0.8 MB fp32, compute well under 0.1 ms, and the graph grows by 4-6 nodes. Generate keys with the env's own encoder (team perspective, plane 15 = side) so they match the referee bit for bit.
  - **Expected effect:** on covered positions it acts like the "strong" rows of the toy, about 0.75-0.88 at T = 1 instead of about 0.15-0.35. That is the only measured way to repair the no-signal positions.
  - **Verify before upload:** P(book move) under the referee config (≥ 20 seeds) on every key, and parity (book never fires) on 400 random non-book positions.
  - **Legitimacy:** this is our own policy output, and the referee's docstring presents noise and temperature as non-levers rather than forbidding sharp outputs. Training-side engines (KataGo, Lc0) get diversity from temperature and noise, not books, but a competition seat's job is strength, not diversity. Still, re-read the contest statement for any rule on hard-coded moves (see Gaps).
- **Value shaping, i.e. a value offset to beat FPU 0: skip it.**
  - The toy shows that lifting all root-child Q values above 0 restores the higher cap (+0.3 shift with Q_best −0.15: P(best) at T = 1 goes from 0.74 to 0.87 with a sharp prior).
  - But the net cannot know the root mover. It sees only the leaf's side to move and `actions_left`, and both sides' leaves appear in our tree. A parity trick (+b at actions_left = 1, −b at actions_left = 2) is consistent only for roots at the first action of a turn.
  - At second-action roots it gives once-visited children +b and deeper-visited children about −b, which favours exactly the junk moves.
  - "Make bad opening moves clearly negative" is better achieved by teacher-labelled opening values (distill) than by hand shaping.

### Gaps
- I did not fetch Chess Programming Wiki pages on opening books, or documentation on how TCEC or engine tournaments assign forced openings. The book-legitimacy reasoning rests on the referee docstring and our ownership of the policy output, not on a cited rule.
- **The contest statement must be checked** for any rule against hard-coded or memorized moves, and for a model file-size cap (the book adds about 0.5-1 MB).
- The toy model has no tree, no turn structure and a synthetic Q spread. Its absolute numbers are illustrative; validate on real positions with `reviews/hv_warm.py`.
- I did not count the opponents' a1-a2 line frequencies in the live replays, so the MAG-side coverage size is an estimate.

## Q4. Combining pipelines: in-graph ensembling, teacher value as az_res target, stacking the prior delta

### Takeaway
Within ≤ 512 initializers and about 3 ms per call, only same-base stacking is affordable: hand_v3 FastNet as a shared base, plus a value or policy delta from distill_v3hb, plus the prior pipeline's policy delta. az_res sits on hand_v2, so mixing it in would mean computing a second base and roughly doubling the trunk, which is unaffordable. distill_v3hb alone is already over the latency budget, so any stacking first needs a smaller trunk.

### Cited Findings
- **Graph budgets:** hand_v3 FastNet is 167 nodes and 225 initializers. az_res_v4 is 503 nodes and 444 initializers (on hand_v2). The live hand_v2 uses 416 of 512 initializers. — `agents/hand/RESULTS.md`, `agents/az/RESULTS.md`, `reviews/hand_v2.md`
- **Latency:** distill_v3hb runs at 6.4 ms/call b8 under load and was validator-accepted on only 1 of 3 tries (dry run 155 ms vs 100 ms for hand_v3). distill_v3 (stopgap base) runs at 3.2 ms. — `agents/distill/RESULTS.md`
- **The prior pipeline's delta** is zero-initialized on hand_v3 FastNet logits with a learnable temperature exp(log_t), and the value is untouched (bit-exact). — `agents/prior/RESULTS.md`
- **Evidence for teacher or Q value targets:** the offline Q-mix experiment, 70% vs 28% outcome-only (`agents/az/RESULTS.md`); NNUE λ = 1.0 runs ([Talkchess](https://talkchess.com/viewtopic.php?p=872983)).

### Inferences
- **Best combination to try, if time allows:** distill_v3hb's value delta and policy delta, plus the prior pipeline's policy delta, both on hand_v3 FastNet.
  - It shares one base. Initializers: 225 + two trunks of about 30-40 each, comfortably under 512.
  - The latency is the problem: shrink both trunks (e.g. 24-32 ch × 2 blocks) and re-time.
  - Test with an arena vs distill_v3hb alone.
  - Small caveat: the prior delta was fitted to visit targets that hand_v3's value produced, so its targets are slightly mismatched to distill's value. Policy improvements from deeper search should mostly transfer.
- **Logit averaging across different bases** (az_res on v2 + distill on v3): needs both bases (about 450 initializers before trunks), so it breaks the cap. Skip it.
- **Teacher value as the az_res target:** yes in principle (Q2). It requires new az_res generations, so it is out of scope today.
- **Prior delta on top of az_res's value:** wrong base (hand_v3 vs hand_v2). The prior delta would need retraining on az_res logits. Skip it.
- **Offline distillation of an ensemble** into one small net (KataGo/Lc0 style) is the clean long-term route. It is out of scope today.

### Gaps
- No timing measurement exists for a two-trunk graph on this machine. The latency figures above are from a loaded machine and may not reflect the validator's environment.

## Q5. Should we use our own live replays (incl. stronger opponents' moves) as imitation data?

### Takeaway
Not as imitation labels, because the volume is far too small and the play quality is unvetted. Use them as a **source of positions** labelled by our teacher (DAgger on the real opponent distribution), especially for MAG-side opening-book coverage and for the positions where we lost Q.

### Cited Findings
- CrazyAra's supervised phase used human games before any RL, and its RL extension needed about one million self-play games (search-snippet figure). — [Czech et al. 2020](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2020.00024/pdf); [TUprints](https://tuprints.ulb.tu-darmstadt.de/13370/)
- Our live sample is 76 protocol games across three match files, at about 35 actions per game. In the latest 13 games, opponents made 47 captures against our 65. — `reviews/hand_v2.md` §2-3
- The distill pipeline's DAgger round (r4: MCTS(student) behaviour, teacher labels) was part of the data mix that improved v2 → v3. — `agents/distill/RESULTS.md`

### Inferences
- Roughly 1.3k opponent actions from mixed-strength teams are too few to imitate, and copying them risks importing their blunders.
- Higher value per minute:
  1. Extract every position from live games, both sides' to-move positions.
  2. Label them with the alpha-beta teacher and add them to the distill/prior training sets with ×3-5 weight. These are the states the live opponent pool actually induces.
  3. Count opponent a1-a2 lines to pick the MAG-side book keys (Q3).
- Collection cost is about 15 min (the scan scripts in `reviews/` already parse the protocols). Labelling about 2-3k positions with the teacher on 1 core takes minutes. It only pays off if a retrain happens before about 16:00.

### Gaps
- I did not measure opponent strength per game, or how often stronger opponents' opening lines repeat. Line repetition is what determines book coverage.
