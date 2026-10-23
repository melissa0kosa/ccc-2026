# Perfect-play ceiling under the fixed referee (az_res4f_v2)

Author: side session -45, for the main session. Written 15:45-16:00 UTC, 2026-10-23.
Scripts (scratchpad, copy if needed): `ceiling.py`, `ceiling_sum.py`, `ceiling2.py`, `simscale.py` in
`C:\Users\melis\AppData\Local\Temp\claude\C--Users-melis-Downloads-cloudflight-coding-contest\2537b5b1-8fcc-4ac8-8fd3-b627f7b7b210\scratchpad\`.
Oracle labels saved for ExIt (f7's plan): `levels/level4/agents/distill/data/oracle800_az_res4f_v2_<match>.npz`
(obs, legal, 800-sim visits, root_q, phase, round, actions_left; 300 positions from 16 live games).

## Bottom line

- **Deep search is worth a lot under this referee:** v2@800 vs v2@100 = 37/40 = 92.5% [80, 97], and v2@300 = 26/40 = 65% [50, 78].
  A policy that captured what deep search knows would be the biggest available lever.
- **For this referee, the ceiling is a policy problem.** Putting the oracle (v2@800) move +10 logits at the root makes the referee pick
  it 99% of the time (97% even under warmup noise and sampling) and removes ~95% of the Q-loss gap. A +2 nudge on the right move gives 85% and
  removes ~72%.
- **The referee's 100-sim search plays WORSE than v2's own prior:** the prior's top-1 matches the oracle 68% of the time, the
  referee's choice only 47%. The prior mass on the right move is ~0.31, so 100 sims with virtual loss and noisy shallow values
  drift off it.
- **Global sharpening doesn't fix that** (offline: hit rate flat at 46-47% for ×1.5/×2/×3, Q-loss −12% at best; arena: 42-54%). It trusts the
  existing ordering more without changing which move is on top. Only better move ordering helps, i.e. search distilled into the policy.
- **The Q-loss gap vs the oracle:** warmup 43%, middlegame 44%, low-material 13%. The last is understated, because the oracle shares v2's endgame blind
  spot. The live losses (review az_res4f_v2.md) are 2/6 K+TR vs K+DE endgames, 3/6 low-odds warmup samples, and 1/6 a middlegame horizon.
- **For the 16:20 slot, if both are borderline: prefer az_res4f_v3_eg over prior_4f2_v1w.** Reasoning in §7.

## 1. Method

- Positions: all 300 of OUR decisions in the 16 live az_res4f_v2 games since 15:19 (`data/matches/matches_az_res4f_v2_a.json`).
  Phases: warmup = actions 1-8 (64), low-material = both sides Kernel + <= 1 piece (76), middlegame = the rest (160).
- Oracle: `mcts.run` with az_res4f_v2, 800 simulations, no noise. Q(a) = root child mean value; best = argmax Q.
  Caveat: the oracle uses v2's own value, so in low-material positions it shares v2's endgame blind spot
  (the gate's lowmat check: sign accuracy 0.79, MAG-lost mean -0.30). Low-material losses are UNDER-stated below.
- Referee decision: `mcts.search_action` with the default `MCTSConfig(simulations=100)` = the exact tournament path
  (warmup noise + temperature sampling on actions 1-8, 4 seeds there; greedy afterwards).
- Q-loss = Q_oracle(best) - Q_oracle(chosen). Q is in [-1, 1], so a Q-loss d costs about d/2 win probability at that decision.
- Counterfactuals:
  - "oracle prior +B": add B to the oracle-best move's logit at the root only;
  - "sharpen k": all logits × k on every call, which is a deployable one-Mul change.

## 2. Where the gap is (vs the 800-sim oracle)

| phase | decisions/game | Q-loss/decision | referee picks oracle move | Q-loss/game | share |
|---|---|---|---|---|---|
| warmup (a1-a8) | 4.0 | 0.085 | 36% | 0.34 | 43% |
| middlegame | 10.0 | 0.035 | 51% | 0.35 | 44% |
| low-material | 4.8 | 0.021 (understated) | 49% | 0.10 | 13% |
| all | 18.8 | 0.042 | 47% | 0.79 | |

- The loss is concentrated: 48 of 300 decisions (16%, about 3 per game) lose more than 0.1 Q, and 14 lose more than 0.2.
- Openings do NOT repeat. Of 64 warmup positions, 55 are distinct; only the start position (8×) and one 2nd action (3×) recur.
  An opening book would cover ~2 positions, so drop it.
- The largest single losses are middlegame 2-action turns, e.g. 15778 a21 (0.71) and a18 (0.41), 16302 a21/a22 (0.23/0.26),
  and the warmup's last action a8 (0.28-0.29 in 3 different games).

## 3. The search makes v2 WORSE than its own prior

| phase | v2 raw prior top-1 = oracle best | referee-100 choice = oracle best | prior mass on oracle best |
|---|---|---|---|
| warmup | 67% | 36% | 0.28 |
| middlegame | 64% | 51% | 0.33 |
| low-material | 76% | 49% | 0.31 |
| all | 68% | 47% | 0.31 |

With only ~0.3 prior mass on the right move, 100 simulations are not enough to confirm it. The first batch
(8 leaves with virtual loss 1, FPU 0) spreads over the top 5-8 priors, and shallow, noisy values then pull the visit maximum
to a different move in about a third of the cases where the prior was right.
(The oracle shares v2's prior, so part of the 68% is agreement by construction. It is still the right yardstick for "what v2
would play with more search".)

## 4. What a perfect prior buys (counterfactual at the root)

| change at the root | Q-loss/decision | referee picks oracle move | Q-loss/game |
|---|---|---|---|
| none (v2) | 0.042 | 47% | 0.79 |
| oracle move +2 logits | 0.0115 | 85% | 0.22 |
| oracle move +10 logits ("perfect prior") | 0.0023 | 99% (warmup 97%) | 0.04 |

- A perfect prior removes about 95% of the gap to the 800-sim oracle, even in the warmup: noise eps 0.15 and T = 1.0/0.5 sampling
  do NOT stop a sharp, correct prior (97% hit). An earlier root-only toy simulation predicted 75-88%; the real referee is better.
- So, for THIS referee, "perfect play" is mainly a policy problem. The value only needs to be calibrated enough not to override a
  correct prior. That's consistent with the team's history: value scalings were neutral (rsv1.25 58.8%, vtemp 52%), while warm×3
  on v2 did nothing (51.9%), because v2's warmup top-1 is right only 67% of the time and sharpening a wrong prior doesn't help.

## 5. Search-scaling ceiling in win rate (v2@S sims vs v2@100)

| matchup (same model, referee config otherwise, colours alternate) | result | Wilson 95% |
|---|---|---|
| v2@800 vs v2@100 | 37/40 = 92.5% (cyan 19/20, mag 18/20) | [80, 97] |
| v2@300 vs v2@100 | 26/40 = 65.0% (cyan 14/20, mag 12/20) | [50, 78] |

- This is the strength a perfectly amortised policy could approach. An evaluator whose prior already names the 800-sim move would play
  roughly like v2@800 inside the referee's 100 sims (§4: 99% hit at the root).
- It's an upper bound on prior-only gains: inner tree nodes would also need the better prior, and the 800-sim opponent also searches deeper on
  the opponent's moves.
- 40 games each, so the intervals are wide, but even the lower bound (80%) dwarfs every candidate so far.

## 6. Deployable search exploitation: global policy sharpening

Same 300 positions, same oracle (reproduced exactly: max |dQ_best| = 0), all logits × k on every evaluator call:

| phase | v2 Q-loss / hit | ×1.5 | ×2 | ×3 | oracle +2 (for contrast) |
|---|---|---|---|---|---|
| warmup | 0.085 / 36% | 0.070 / 38% | 0.064 / 40% | 0.066 / 40% | 0.034 / 66% |
| middlegame | 0.035 / 51% | 0.031 / 51% | 0.035 / 49% | 0.041 / 47% | 0.006 / 91% |
| low-material | 0.021 / 49% | 0.021 / 47% | 0.018 / 49% | 0.018 / 49% | 0.006 / 91% |
| all, Q-loss per game | 0.79 | 0.69 | 0.69 | 0.76 | 0.22 |

Reconciliation with the arena (deriv agent: global ×1.25/×1.5 42-54%, rsp125 48/100, R1×6 52%, warm×3 51.9%):
- Both metrics agree that sharpening is ~neutral. The hit rate is flat, and the Q-loss gain (≤0.1 Q/game, mostly in the warmup) is below what
  100-300 arena games can resolve.
- Mechanism: where the referee leaves v2's top-1 move, the shallow Q of that move looks worse, and ×3 logits still don't outweigh it. Where
  v2's top-1 is wrong (32%), sharpening entrenches the error. Only a prior that ranks the RIGHT move first (the +2 column) closes the gap.
- The offline metric is circular in one direction: the oracle shares v2's prior and value. It can't credit fixes to v2's blind spots
  (endgames) and may over-credit "agree with v2@800" changes. The arena vs v2 has the mirror bias (v2 is the opponent). For
  endgame fixes, trust the live-loss review and the lowmat suite, not this oracle.

## 7. Recommendation for the last upload slot (16:20)

**If both are borderline, ship az_res4f_v3_eg rather than prior_4f2_v1w.**

1. v3_eg targets the losses that actually happen and uses the only lever with a validated large arena effect:
   - endgame data took 4f_v1 → 4f_v2 from +0 to +15 pp;
   - the v2 review shows v3 flips the cliff values (15273 a35 +0.33 → −0.18, a36 +0.23 → −0.43, 16438 a31 +0.38 → −0.16; lowmat sign-acc 0.83 → 0.86);
   - the exact override (rule a) fixes the bare-Kernel suicide-capture pattern, with bit parity elsewhere.
   - These are errors the v2@800 oracle can't see, so my Q-loss table under-credits v3_eg.
2. v1w is the right mechanism (policy distilled from 800-sim search), but this build doesn't deliver it yet:
   - its held-out top-1 agreement with the 800-sim move improved only in the warmup (0.607 → 0.679) and FELL post-warmup
     (0.659 → 0.615), where 44% of the Q-gap is (prior/RESULTS.md);
   - its labels come from v2@800, which shares the endgame blind spot;
   - its 63% comes from the prior agent's arena, which read prior_4f2_v0 at 56% while the gatekeeper read 47%. Expect ~54% at the gatekeeper.
   - The warmup losses it targets were ≤3%-odds samples (review): small expected gain per game.
3. Tie-break rule: if the gatekeeper numbers differ by under ~5 pp, take v3_eg (independent evidence from live losses). Take v1w only if it beats
   v3_eg by a clear margin in the same arena.

## 8. Post-contest: how to approach perfect play under this referee

1. Amortise deep search into the policy (ExIt). Label tens of thousands of positions with ≥800-sim visit distributions (start from
   `agents/distill/data/oracle800_*`), train the residual policy to rank that move first by ≥2 logits, and iterate.
   - The ceiling says this could approach v2@800 strength (92.5% vs v2@100).
   - 2.76k labels were too few: post-warmup agreement didn't improve.
2. Replace the oracle's value where it's blind: exact endgame tablebase values (finite-horizon backward induction; see the
   round-3 report), so the labels are right exactly where v2@800 is wrong.
3. Keep the value calibrated, not sharp. Once the prior is right, the value's only job is to avoid overriding it (§4).
4. Measure with both instruments: oracle hit rate / Q-loss on fixed positions (cheap, low noise) AND a gauntlet. A head-to-head vs the parent
   alone shares the parent's biases.
