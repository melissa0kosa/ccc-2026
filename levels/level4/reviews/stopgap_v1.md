# Review: stopgap_v1 (live ~13:51-14:01 UTC, 2026-10-23)

Scratch scripts and data are in `levels/level4/reviews/`: `sg_stats.py` (end reasons), `sg_pos.py MATCH K... [--mcts] [--board]` (model value and prior at a replayed position), `sg_scan.py` (`sg_scan.txt`/`.json`: each of our decisions re-searched with hand_v2 MCTS-400; 11 games, without 4869), `sg_fw.py` (round-8 evacuation test), `sg_warm.py` (sampling odds in 4869), `sg_verify.py` and `sg_cpu.py`.
Action numbers below are protocol numbers ("a29" is protocol line 29, the 29th action).

## 1. Verification: PASS (validator is flaky only because the machine is loaded)
- `c1.py validate`: the first 2 runs were REJECTED with "too slow to check" (the 2 s CPU / 3 s wall check failed while other agents were training). The 3rd run was **ACCEPTED**. The model itself is tiny, so these were load failures, not model failures.
- Contract: input `observation [batch,16,8,8]` float32; outputs `policy [batch,4096]` and `value [batch]` float32.
- On 1009 real live positions, every output is finite. Value lies in [-0.99967, +0.99997] (inside [-1,1], but saturated; see W2). Logits lie in [-0.5, 6.5].
- CPU with the referee's 1-thread OnnxEvaluator: 0.21 ms per call at batch 1, 0.57 ms at batch 8. Self-play at 100 sims costs 1.65 s CPU per game for both sides (max 3.0 s over 6 games, under load), far below the 15 s match limit.

## 2. Live performance
Window: `played_at` >= 13:51 in `matches_1405_stopgap_v1.json`, i.e. **12** games (13:51:21-13:59:58). The file was downloaded just before hand_v2 went live, so this is the full stopgap_v1 sample.

| model | games | wins | as CYAN | as MAGENTA |
|---|---|---|---|---|
| stopgap_v1 | 12 | **10 (83%)** | 4/5 | 6/7 |
| seat_l3_material (previous, `tournament-matches.json`) | 18 | 7 (39%) | 0/9 | 7/9 |

End reasons:

| | stopgap_v1 | seat_l3_material |
|---|---|---|
| Win: captured their Kernel | 9 (PI 2, KE 2, TR-shot 2, IC 2, EX 1) | 3 |
| Win: bare-Kernel rule | 1 (5530) | 0 |
| Win: their Kernel burned by firewall | 0 | 4 (3430, 3912, 3945, 4044*) |
| Loss: our Kernel captured | 2: 5471 (CYAN; by KE, after the firewall burned our Daemon), 4869 (MAGENTA; by PI, which first took our only guard, then the Kernel; a10) | 7 |
| Loss: our Kernel burned by firewall | **0** | 4, all as CYAN (3655, 3786; 3868 and 4012 burned both Kernels, which counts as a MAGENTA win) |
| Own non-Kernel pieces burned by firewall | **4 pieces in 3 of 5 CYAN games** (5051 TR; 5079 TR+IC; 5471 DE), 0 as MAGENTA | many |

\*4044: both Kernels burned while we were MAGENTA, so the game went to us.

stopgap_v1 fixed the previous seat's main leak (its own Kernel burning in the round-8 shrink as CYAN). What remains: non-Kernel material burning as CYAN (W1), and a single-guarded Kernel falling to a 2-action strike (W5).

## 3. Weaknesses, ranked (with evidence)
Ranked by live damage: **W1 (loss 5471, 4 burned pieces) > W5 (loss 4869) > W2 (amplifies both) > W3 > W4 > W6.** Each weakness keeps its number below.
For the reference column, each of our decisions was re-searched with hand_v2 at MCTS-400 (`sg_scan.txt`). Caveat: this reference is biased toward hand_v2.

### W1. CYAN does not evacuate non-Kernel pieces before the round-8 shrink. This caused loss 5471.
- **Evidence: 5471, a29-a30 (CYAN, round 8).** Our Daemon stands on C1 (outer ring). stopgap played KE C3->D4 and TR C4->C5. The reference gives Q +0.70 for saving the Daemon (best C1->F1) against +0.01 for the move played. The Daemon burned (`FIREWALL ROUND 8: [CYANxDE (C1)]`), and at a33 every move already scored Q <= -0.69: the game was lost there. The later Kernel walk (a33-34, KxIC on F5) only finished it off.
- **The same pattern** appears in 5079 a29-a30 (stopgap shuffled PI C7->B7->C7 while TR A3 and IC A2 burned) and in 5051 a29 (IC F2->F4, F4->E4 while TR A3 burned).
- **Controlled test (`sg_fw.py`):** play CYAN's whole round-8 turn with MCTS-100 over 4 seeds. stopgap left material on the ring in **12/12** runs (D, then TI, then T). hand_v2 evacuated everything in **12/12** runs: `C1->C2+C4->B4`, `A2->C4+A3->B3`, `C2->C3+A3->B3`.
- **Cause in stopgap.py:**
  - Value: `w_danger * urg * danger * (own_k - en_k)` scores **only Kernels**. No value term counts own non-Kernel material on the next ring.
  - Prior: `from_term = prior_escape * own_k * danger * urg` gives an escape bonus to the Kernel only. `to_term` merely penalises stepping *into* fire.
  - Timing: the shrink fires after MAGENTA's turn. For CYAN the material loss therefore shows up only 3 or more plies below the root, beyond what a flat-prior 100-sim search reaches. For MAGENTA the shrink is applied at its own turn end, so its search sees it; that explains 0 burns as MAGENTA.
- **hand_v2: FIXED.** It has the features `ownV_N1_u1`, `enV_N1_u1_mag/_cyan` and `ownV_N1_u` plus the policy terms `fw_from` and `fw_to`. At 5471 a29 its prior is C1->C2 0.38, against 0.05 for stopgap.
- **Fix if still needed:** keep these hand_v2 terms. Optionally add `ownV_N1_u1 * (1-mag)` as a separate feature: CYAN's last chance is its own round-8 turn, while MAGENTA's view already contains the shrink one ply later.

### W2. The value is saturated, so search cannot tell good from bad
- **Evidence:** in 5471 a21-a30, stopgap's value is +0.98 to +0.99 while the reference root is -0.04 to +0.59 (and a33 turns out lost). In 5079 a28 it is +0.995. Over our 206 live decisions, |v_sg| > 0.9 in **23%** (hand_v2: 13%). The correlation with the reference root value is **0.59** (hand_v2: 0.86), the MAE is 0.34 (0.15), and the sign agrees in 73% of positions (84%).
- With v pinned near ±1, every child's Q looks the same. Search then picks among null moves (the PI shuffle in 5079 a29-30; the KE C6<->D6 shuffle in 5079 a41-a61) instead of finding the move that keeps material.
- **Cause:** `value = tanh(v_scale * v)` with `mat = 1/40` on VALUES up to 50, so one Daemon is worth 1.1 before tanh. On top of that come the `w_danger`, `w_center`, `w_guard` and `w_thr` terms, and all of them ignore planes 13 and 15.
- **hand_v2: MOSTLY FIXED** (its weights were Texel-fitted). It still reaches |v| > 0.9 in 13% of positions.
- **Fix (15 min):** sweep a global temperature on hand_v2's tanh input (multiply the `lin` output by 0.7, 1.0 and 1.3) in a 40-game arena against hand_v2 itself. If refitting, add a calibration check: the mean |v| per round bucket should track the observed win rate.

### W3. No 2-action threats, and the capture prior ignores whether the landing square is safe
- **Evidence: 5471 a34-a35.**
  - a34: CYAN KE E4->F5 captures the Ice. The landing square is Chebyshev distance 2 from the enemy Kernel (G7), and the enemy Ping can strike it with a 2-action sequence (F3->D3, then D3xF5).
  - a35 (MAGENTA to move, forced win in 2): stopgap values it **-0.72** for MAGENTA, the wrong sign. hand_v2 values it **+0.97**, which is correct.
  - stopgap's prior gave E4->F5 0.41 at a34, from the capture bonus `prior_cap * en_val` with no safety check.
  - Note: the position at a33 was already lost, so this misjudgement did not decide the game. It shows the blind spot.
- **Cause:** `reach()` kernels cover a single action only: Kernel `d==1`, Ping diagonal <= 2 from its current square. Yet a turn has 2 actions, a Kernel walks 2 squares, and a Ping can make a quiet move and then capture. `w_thr` and `w_hang` see only 1-action attacks, and `to_term` has no danger factor for the moving piece.
- **hand_v2: VALUE FIXED, PRIOR NOT FIXED.** The value is fixed through the `t2_own`, `t2_en`, `win2` and `ownK_thr` features. The prior is not: hand_v2 still puts **0.92** on E4->F5 at a34, because its prior `danger` map uses the 1-action `hit_en` only.
- **Fix for hand_v2 (15 min):** in `policy()`, build a 2-action enemy attack map:
  ```
  hit2_en = clamp(conv(_mv_en (per type) + en (per type), unblocked 7x7 reach kernels per type), 0, 1)
  ```
  stopgap's `reach()` can supply the kernels. Then add the term `- P["k_danger2"] * ownK(from) * hit2_en(to) * (g_own==0)` with weight about 3. Optionally add `- 0.3 * ownV(from) * hit2_en(to)` for the other pieces. Re-arena against hand_v2.

### W4. The prior is flat in quiet positions, so 100 sims stay shallow
- **Evidence:** in quiet positions stopgap's top prior is 0.04-0.08 spread over 15-30 legal moves. Examples: 5051 a17 (0.04 over 29 moves), 5079 a37 (0.08 over 14), 5471 a29 (0.05 over 23).
- Each root move then gets about 3-6 visits, so the search sees about 2 plies, too shallow for CYAN's 3+-ply firewall horizon (W1) or for opponent 2-action strikes (W3).
- **Cause:** `to_term` is non-zero only on enemy-occupied or danger squares. `from_term` is non-zero only for a Kernel in fire. Nothing raises the prior of quiet moves that matter: evacuation, guarding, approaching the enemy Kernel.
- **hand_v2: PARTLY FIXED.** It concentrates the prior on firewall evacuation (0.34 + 0.28 at 5079 a29) through its `esc`, `near_enK` and `center` terms. In quiet middlegames it is still flat (5079 a40: top 0.08).
- **Fix (15 min):** add a from-term `+0.5 * ownV * hit_en * (1 - hit_own)` for every piece type, not only the Kernel (hand_v2's `esc` already does this). Add a to-term `+0.3 * adj_ownK * empty` (guard your own Kernel). Then try `p_scale` around 1.5 so that the top 3 moves hold about 60% of the prior.

### W5. The bodyguard logic counts one guard as full protection. This caused loss 4869.
- **Evidence: 4869 (MAGENTA), a7-a8, round 2.** After a6 (cyan PI D4xF6, taking our Daemon), the cyan Ping on F6 attacks our Ice G7, the **only** guard of our Kernel H8. Its line goes F6xG7, then G7xH8.
- **stopgap's evaluation.** stopgap gives a7 and a8 v = -0.93 / -0.94 (saturated, same as a quiet material deficit) and a flat prior: the saving capture IC G7xF6 gets 0.08, the top prior but nearly tied.
  - a7: the move played was PI G4->E4 (prior 0.04, rank 17/25), sampled during warmup at T=0.5. Over 10 seeds, the search visit share of G7xF6 was 0.61 on average.
  - **a8 (one action left, decisive):** stopgap's visit share for G7xF6 is only **0.15**, so P(sample) is about 0.42. The move played was PI E4->E3.
  - Then cyan played a9 PxIC G7 and a10 PxKE H8.
- **hand_v2 at a8:** visit share 0.81 and P(sample) 1.00, so **FIXED**.
- **Cause in stopgap.py:**
  - `own_guard = clamp(adj, 0, 1)` makes one guard count as safe, and the `(1 - own_guard)` factor zeroes `w_thr` on our Kernel. The unblocked Ping reach really does put F6 on H8, but the term is switched off because the Ice is adjacent.
  - The attacker keeps its square after a redirect, and a direct capture of the guard is just as good. With 2 actions, therefore, **one guard under attack means the Kernel is lost**.
  - `w_guard` even pays +0.35 for this guard.
- **hand_v2:** FIXED through `g0`/`g1` combined with `a2` in `win_now`, `ownK_thr` and `t2_own`, plus the prior's `esc` term on the attacked guard.
- **Fix if a stopgap-style net is kept:** count guards that are not attacked: `guard_safe = own_nk * (1 - en_att)`. Use `clamp(adj(guard_safe) - 1, 0, 1)`, which requires 2 safe guards before the threat term is zeroed. Add `- w_thr * own_k * en_att2` for squares attacked in 2 actions (see W3).

### W6. Reach maps ignore blocking and the side-to-move planes (no decisive live evidence)
- The unblocked 7x7 reach lets Daemon, Ice and Trace "attack" through pieces, and ignores the Trace's adjacent-diagonal block. Planes 13 (actions left) and 15 (MAGENTA wins tiebreaks) are unused.
- The kind of evidence available: 5079 a41-a61 shows an equal K+P vs K+P endgame valued +0.57 to +0.92 for CYAN, although MAGENTA wins equal-ring tiebreaks.
- **hand_v2: FIXED.** Its slides are blocking-aware (shift convolutions masked by `empty`), and it has the `a2`, `mag`, `mag_late` and `Kdepth_diff_bare` features.

### Not model faults
The flagged MAGENTA opening moves (5640 a3, 5590 a8, 5236 a3, 5111 a4, 5066 a7-8) fall inside the referee's warmup window (temperature sampling plus Dirichlet noise), so they are not model faults.

## 4. Bottom line
- stopgap_v1 was valid and cheap. Over 12 live games it scored 10/12, against 7/18 for seat_l3_material.
- Its 2 losses came from W1 (5471: a Daemon left on the ring at CYAN's round-8 turn; the same leak burned 4 pieces in 3 of 5 CYAN games) and W5 (4869: a single guard counted as safe against a 2-action Ping strike).
- hand_v2 fixes W1, W5 and W6, and mostly fixes W2 and the value side of W3.
- Highest-value remaining work for hand_v2: the **2-action danger map in the prior** (W3) and a **value-temperature sweep** (W2).
