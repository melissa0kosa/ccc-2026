# Review: hand_v2 (live since ~14:01 UTC, 2026-10-23)

Sample: `matches_hand_v2_1.json`, entries with `played_at` >= 14:01:30, i.e. 13 games with a protocol (14:03:11-14:14:08), plus 2 entries without a protocol (see section 3).
Action numbers are protocol line numbers ("a23" is the 23rd action).

Scratch scripts and outputs are in `levels/level4/reviews/`:
- `hv_stats.py`: end reasons.
- `hv_verify.py` -> `hv_verify.txt`: latency, value and prior statistics, CPU.
- `hv_scan.py` -> `hv_scan2.txt` / `hv_scan2.json`: each of our 216 decisions searched at MCTS-100 and MCTS-800, both greedy with no noise and the referee's tie-break.
- `hv_warm.py`: referee warmup sampling odds, plus the sharpened-prior test.
- `hv_feat.py`: hand_v2 features at critical positions.
- `validate_hand_v2.txt`.

Ignore `hv_scan.txt`/`.json`: that first run used a different tie-break.

## 1. Verification: PASS
- **Validator:** `c1.py validate` was **ACCEPTED on the first try** (112.6 ms dry run; one Level 3 match of 18 actions vs Greedy).
- **Graph:** 469 nodes (cap 1024) and **416 initializers (cap 512, 81% used)**. Opset 17, 107 KB. I/O is `observation [batch,16,8,8]`, `policy [batch,4096]`, `value [batch]`.
- **The live seat runs exactly this file.**
  - With the referee's tie-break (`select_action_index`: equal visits go to the lowest action index), local hand_v2 MCTS-100 reproduces **164/164** of our post-warmup live moves.
  - With Python's `max(N)` tie-break, 8 of them "differ". All 8 are exact visit ties; for example 6032 a16 has F5->F7 and F5->F4 at 11 visits each, and the referee path picks F5->F4 in 40/40 runs.
  - The torch source (`hand_v2_src.py` + `w_v2.json`) gives the same value as the ONNX to 3 decimals at the probed positions.
- **Outputs on 426 live positions (both sides):** all finite. Value lies in [-0.937, +0.9996], with |v| > 0.9 in 7.5% and mean |v| 0.40. Logits lie in [-4.8, 10.3].
- **Latency** (referee `OnnxEvaluator`, 1 thread): session creation 94 ms; **0.51 ms per call at batch 1 and 1.02 ms at batch 8**.
- **CPU per game:** no sign of timeout risk.
  - Own-side 100-sim search: mean 33-35 ms, max 47 ms, 13-14 evaluator calls per search.
  - Our 3 longest live games (48, 42 and 42 actions) cost 0.70-0.80 s of own CPU.
  - A maximum game of 104 actions means 52 own searches: about **1.8 s, worst case 2.4 s**, against the 15 s limit (more than 6x headroom).
  - Self-play, both sides: 1.1-2.1 s per game (31-63 actions).

## 2. Live performance

| model | games | wins | as CYAN | as MAGENTA | own pieces burned by firewall | mean length |
|---|---|---|---|---|---|---|
| **hand_v2** (>= 14:01:30) | 13 | **10 (77%)** | 5/6 | 5/7 | 1 piece (6302 TR A2, CYAN, game won) | 33 actions |
| hand_v2 incl. the 2 protocol-less entries | 15 | 10 (67%) | 5/8 | 5/7 | | |
| stopgap_v1 (previous seat) | 12 | 10 (83%) | 4/5 | 6/7 | 4 pieces in 3 of 5 CYAN games | |

The 10/13 vs 10/12 difference is noise (Wilson 95% CI for 10/13 is [50%, 92%]). The opponent pool has also changed: the rating climbed, so opponents are stronger.

**How the games ended**
- **Wins (10):** all by Kernel capture. Capturing piece: TR shot 3 (5994, 6110, 6815), IC 3 (6294, 6585, 6754), DE 2 (5905, 6607), KE 1 (6302), PI 1 (6470). There were no bare-Kernel or firewall decisions.
- **Our captures:** 65 in total (including the 10 Kernels). The opponents made 47.

**Losses (3, all by Kernel capture):**

| match | colour | end | how it was lost |
|---|---|---|---|
| 6177 | MAG | a22 PI x KE | Warmup blunder at a7 (W1). The CYAN Exploit then took IC (a6), DE (a9) and TR (a10). |
| 6509 | CYA | a36 DE x KE | Warmup blunders a2 and a6, then 2-action strikes cost EX (a4) and DE (a8) (W1+W2). Bare Kernel vs K+DE from a26. |
| 6032 | MAG | a42 KE x KE | Played a23 IC G7->E5, which walked into a Kernel 2-walk recapture (W2+W4). At a36 the lone guard was lost to a quiet move + Trace shot (W3). Bare Kernel vs K+TR from a38. |

**Fixed relative to stopgap_v1:**
- Round-8 evacuation: 1 burn in 13 games, against 4 pieces in 3/5 CYAN games before.
- Single-guard Ping strikes (stopgap W5): none seen.
- No Kernel was ever burned.

**Near-disasters in won games:** none serious. The lowest MCTS-800 root value in a won game was -0.45 in 6110 (a9, round 3, recovered) and -0.27 in 6585 (a22).

## 3. The two protocol-less entries (5764 at 14:01:28 and 5806 at 14:01:54, CYAN, score 0)

**What they are:** most likely forfeits recorded during the model swap, not anything hand_v2 did in a game.

The evidence that they count as losses for us:
- In every one of the 76 protocol games across all three match files, `score` is 1 exactly when CYAN won. The field is from CYAN's perspective, so `score 0` with `you=CYAN` is recorded as a **loss for us**.
- Both entries have `opponent: None`, while all 28 protocol games have a string. Both have an empty protocol and the standard starting observation.

Why they are tied to the deploy:
- Both fall in the deploy gap: the last stopgap game was at 13:59:58, these two at 14:01:28 and 14:01:54, and the first hand_v2 protocol game at 14:03:11.
- CYAN moves first, so a seat that could not load or evaluate its model at the first call produces exactly this: no moves, CYAN loses.

Why it is not a CPU timeout:
- A timeout needs a long game and would leave a partial protocol.
- hand_v2 needs about 35 ms per search and 94 ms for session creation.
- It has since played 13 clean games, up to 48 actions long.

**Caveat for the upload gate:** the stopgap_v1 swap at 13:51 left no such entries; that match file spans 13:39-13:59. So this is one observation, not a confirmed per-upload cost. If the server does count them, each upload risks about 2 free losses. Factor that into the "bundle small gains" rule.

## 4. Weaknesses, ranked by live damage (with evidence and a ~15 min fix each)

### W1. The opening warmup samples blunders from flat visit distributions (loss 6177; contributed to 6509)
In actions 1-8, the referee samples with T = 1.0 and then 0.5 (proportional to visits^(1/T)) and adds Dirichlet noise. hand_v2's warmup prior is flat: median top-1 0.06. The median visit share of the greedy move is only 0.17.

**What the warmup costs:**
- In 30 of 52 warmup decisions, the move played was not the greedy-100 move.
- Mean Q-loss (MCTS-800) per warmup decision is **0.079**, against **0.035** after the warmup.
- 7 warmup decisions lost more than 0.2: 6177 a7, 6110 a2, 5905 a4, 6302 a1, 6470 a3, 6294 a7, 6607 a7.

**6177 a7 (MAG):**
- The CYAN Exploit on F6 forks DE G8 and TR H6.
- G5xF6 is the prior top-1 (0.39) and the choice of both MCTS-100 and MCTS-800 (Q +0.03).
- PI G5->G4 was sampled instead: prior 0.02, Q -0.33, and **0/40 in a re-run of the referee config**, so it was rare bad luck.
- Exploit chain a9-a10 followed, and the game was lost.

**6509:**
- a2 DE C1->C4: sampled with P = 0.03. It left EX E4 to the strike TR H6->H7, H7xE4 at a3-a4.
- a6 DE C4->C7: P = 0.07. It was taken by EX E7->D5, D5xC7 at a7-a8.
- From there the position was about -0.7.

**Cause:** the prior has no signal in quiet opening positions, so the visits stay flat and sampling roams.

**Fix (5 lines in `policy()`):** sharpen the logits only inside the warmup window. The model can detect it: rounds 1-2 are the only rounds with `u <= 0.25` and no locked ring.
```
warm = clamp((0.375 - u) * 8, 0, 1) * (1 - clamp(nlock, 0, 1))   # 1 in rounds 1-2 only
logits = logits * (1 + 0.5 * warm)
```
- `u = x[:,14,0,0]`; pass `nlock` from `features()` into `m`.
- Measured at 6177 a7 with the referee config over 40 seeds: P(G5xF6) is **0.78 now, 0.97 at x1.5, 1.00 at x2**.
- It does not help at 6509 a2/a6, where the prior has no signal. Those need W2.
- Post-warmup search is unaffected.

### W2. 2-action material strikes are invisible to both value and prior (loss 6032; 6509 a3-a4 and a7-a8; 5471 a34 from the previous review)
`own_hang_*`, `en_hang_*`, `en_att2_*` and the prior's `danger`/`danger_def` term all use the 1-action maps `hit_en`/`hit_own`. Only the Kernel has 2-action features (`t2_*`, `win2`).

**Evidence: 6032 a23 (MAG).**
- IC G7->E5 puts the Ice at Chebyshev distance 2 from the CYAN Kernel on C3.
- MCTS-100 chose it. MCTS-800 says Q **-0.40 vs -0.19** for H6->G6 (dQ 0.21, among the 9 largest gaps).
- After a24, `hv_feat.py` shows CYAN to move with 2 actions while the Ice is capturable by KE C3->D4->E5. Yet `en_att_I = en_hang_I = 0`, and CYAN's value is only +0.32.
- CYAN played exactly that at a25-a26.

**Fix (15 min):**
- **Kernel-walk zone (minimal version):** `kz2_en = clamp(conv(enK, ones(5,5), pad=2), 0, 1)` (and the mirror for us).
  - Value features: `own_hang2_<X> = S(own_X * kz2_en * (1 - hit_own))` for X in P, D, X, T, I. Start with weight about -0.35 x the `own_<X>` weight, or refit with `fit.py`.
  - Prior: add the from-to term `- P["danger2"] * ownV(from) * kz2_en(to) * (1 - hit_own(to))`, with weight about 0.4.
- **General version:** `hit2_en = clamp(Σ_t conv(_mv_en[:, t], REACH_t) + kz2_en, 0, 1)` with unblocked reach kernels (stopgap's `reach()`). This also covers the Ping and Trace strikes in 6509 and 5471.

### W3. A single guard under a 2-action Kernel strike costs the guard, and the value does not see it (loss 6032)
**Code:** `thr_own = g0*clamp(ka+t2o) + g1*ka` counts 2-action strikes only on unguarded Kernels. With one guard, a quiet move followed by an attack sacrifices the guard. No feature carries this:
- `t2_own` was fitted to weight **0.001**, and `t2_en` to **-0.03**.
- The priority order is Ping, Exploit, Trace, Daemon, Ice, so a lone guard always dies.

**Evidence: 6032 a36 (MAG).**
- TR G7->G6 leaves KE F7 guarded only by that Trace, on the diagonal of the CYAN Trace on C3.
- MCTS-800: Q **-0.24 vs -0.10** for F7->E7 (100-sim chose G7->G6).
- `hv_feat.py`: after G7->G6, `t2_en = 1`, `g_en = 1` and `enK_thr = 0`, so CYAN's value is +0.278, against +0.292 after F7->E7. The value is blind to it.
- a37-a38: C3->C4, then C4 shoots F7 and our Trace dies. That left a bare Kernel against K+TR, lost at a42.

**Fix (2 lines + weights):** add `ownK_g1_t2 = g1_own * t2o` and `enK_g1_t2 = g1_en * t2e`, hand-set to about **-0.4 / +0.4**, or refit. Optionally scale by the cheapest guard's value.

### W4. The Kernel-attack prior ignores guards (item (i) of the previous review; 6032 a24, 5471 a34)
**Code:** in `policy()`, the to-square term `cap * enV` uses `enV = enV + enK*ek_val` (cap 1.34, ek_val 4.37), so any attack on the enemy Kernel square gets a logit of about 5.9. That happens even when the Kernel is guarded and the only effect is that the priority guard dies while the attacker stays on its origin square.

**Evidence:**
- 6032 a24: E5->C3 at prior **0.98**. It killed EX C4, and the IC left on E5 was recaptured.
- 5471 a34: prior 0.92.

**Fix:** `enV_to = m["enV"] + m["enK"] * (P["ek_val"] * g0_en + 1.5 * (1 - g0_en))`, where `g0_en = relu(1 - g_en)` broadcast per batch (`g_en` is already in `m`). Optionally subtract a `danger(origin)` term for the attacker. This makes the 0.98 prior about 0.2-0.3 and lets the search look at alternatives.

### W5. The prior is flat in quiet positions, and the Kernel is reluctant to step off the next ring early
**Prior flatness (426 live positions):**
- Top-1 percentiles: p10 0.05, p25 0.07, **p50 0.115**, p75 0.30, p90 0.84.
- 46% of positions have top-1 < 0.10 (median 21 legal moves); median top-3 mass is 0.29.
- With two actions left, the median top-1 is 0.085.

**The 800-sim choice is often a low-prior move:**
- It is the prior's top-1 in only 40% of decisions; its median prior is 0.08.
- Several of the largest gaps are the Kernel stepping off the corner/outer ring in rounds 6-7:
  - 6585 a26: A1->A2 at Q +0.65 vs +0.33 played.
  - 6754 a22: A1->A2 at +0.68 vs +0.47.

**Cause:** `k_from = -0.22` taxes every Kernel move, and `fw_from`/`k_fw` only fire when `u1` (shrink this round).

**Fix:** add `+0.6 * ownK * N1 * u2` to the from-term (`u2` = shrink within 2 rounds). Longer-term, refit `pol` with `fit.py --pol` on 800-sim visit targets.

### W6. Minor: endgame conversion and value saturation
**Slow conversion in won games:**
- 6585 a33 and a37: the same IC C3->C5 shuffle twice, Q +0.06 vs +0.38 for C3->D4. The game dragged to a42.
- 6815 a31: missed a forced win, Q +0.66 vs +1.00 for F6->E6 (prior 0.02).

**Saturation is fine:** |v| > 0.9 at 7.5% of live positions (10.6% of our decision points), mostly in decided positions. Correlation with the MCTS-800 root value is **0.93**, MAE 0.14. Not a priority.

## 5. 100-sim vs 800-sim disagreement (hand_v2 on both sides, greedy, referee tie-break)

| decisions | 100 != 800 | median dQ | dQ > 0.1 | dQ > 0.2 | dQ > 0.3 | mean Q gain per decision |
|---|---|---|---|---|---|---|
| all 216 of ours | **77 (35.6%)** | 0.057 | 24 (11%) | 9 (4.2%) | 5 (2.3%) | 0.030 |
| post-warmup 164 | 57 (34.8%) | 0.072 | 24 | 9 | 5 | 0.035 |
| warmup 52 | 20 (38.5%) | 0.042 | 0 | 0 | 0 | 0.015 |

**Reading:**
- About a third of decisions would change with 8x more search, but most of those are near-ties.
- Only about 4% of decisions are improvements worth more than 0.2 Q.
- The deeper search mostly gains by finding low-prior moves (W5), which argues for prior fixes over value fixes.
- Caveat: the reference is hand_v2 itself, so it shares the blind spots of W2/W3. 6032 a36 shows only dQ 0.14 at 800 sims, although that move lost the game.

## 6. Bottom line
- hand_v2 is valid, reproducible (164/164 live moves) and cheap (at most 2.4 s of CPU per game against 15 s). It scored **10/13 (77%)**, with balanced colours: 5/6 as CYAN, 5/7 as MAGENTA.
- The 2 protocol-less CYAN entries are swap-time forfeits (score 0 = recorded CYAN loss), not timeouts.
- **Losses:**
  - 6177: warmup sampling (W1).
  - 6509: warmup sampling plus 2-action strikes (W1+W2).
  - 6032: Kernel-walk recapture plus a lone guard lost to a quiet move + shot (W2/W4 + W3).
- **Best ~15-minute bundle, lowest risk first:**
  1. **W4** (guard-aware Kernel-attack prior).
  2. **W3** (`ownK_g1_t2` / `enK_g1_t2` features).
  3. **W2** Kernel 5x5 walk zone in value and prior.
  4. **W1** warmup-only logit sharpening (x1.5).

  Arena the bundle against hand_v2 under the upload gate (>= 60 games).
