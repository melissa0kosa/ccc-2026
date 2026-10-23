# Review: az_res_v6 (live 14:43-14:53 UTC, 2026-10-23), plus the first az_res_v20 losses

Sample: `matches_az_res_v6.json`, played_at 14:43:30-14:53:30: 11 games, all with protocol, no protocol-less (swap) entries.
v20 addendum: `matches_az_res_v20_a.json` >= 14:53:30 (6 games; losses 11686, 12113, 11791).
"aN" = protocol action N. Q800 = Q of a move after an 800-sim search (greedy, no noise, referee tie-break), from the mover's side.

Scripts and outputs in `levels/level4/reviews/`:
- `v6_common.py` (window loader), `v6_stats.py` (end reasons), `v6_end.py` / `v6_boards.py` (game endings and boards)
- `v6_verify.py` -> `v6_verify.txt`, `validate_az_res_v6.txt`
- `v6_scan.py` -> `v6_scan.txt` / `v6_scan.json` (all 219 of our decisions, MCTS-100 vs MCTS-800), `v6_stats2.py` (summary)
- `v6_probe.py` / `v6_probe2.py` -> `v6_probe.txt` / `v6_probe2.txt` (critical positions: v6, v20, hand_v4f)
- `v6_lowmat.py` (raw-value calibration in low-material positions, v6 + v20 games)

## 1. Verification: PASS (live-verified; local validator only fails under load)
- **Live seat runs exactly this file:** local v6 MCTS-100 reproduces **175/175** post-warmup live moves. The file equals `rollback_az_res_v6.onnx` (sha256 deb67cc5...).
- **Graph:** 503 nodes (cap 1024), **444 initializers (cap 512, 87% used)**, opset 17, I/O `observation [batch,16,8,8]` -> `policy [batch,4096]`, `value [batch]`.
- **Validator:** ACCEPTED earlier (RESULTS.md, 14:3x). Re-run now 5/5 REJECTED, every time with "too slow to check: 2 s CPU / 3 s total". 42 python processes were running on the box. That is the known load artefact (RESULTS.md, 14:40: hand_v2fast and v6 were rejected the same way). The server played 11 games with it, the longest 94 actions (10667), with no forfeits. Re-validate on a quiet machine before any further upload.
- **Outputs on 431 live positions:** all finite. Value lies in [-0.9994, +0.9999]. **|v| > 0.9 in 20.9%** (hand_v2: 7.5%), mean |v| 0.53. Logits lie in [-6.5, 12.8].
- **Latency (1 thread, under heavy load):** session creation 125 ms CPU, **0.94 ms per call at batch 1, 2.81 ms at batch 8** (hand_v2: 0.51 / 1.02). Arena CPU is 1.3 s per game for both sides, so a 104-action game stays far below 15 s.

## 2. Live stats

| model | games | wins | as CYAN | as MAGENTA | caps us/them | end of losses |
|---|---|---|---|---|---|---|
| **az_res_v6** | 11 | **8 (73%)** | 5/6 | 3/5 | 57 / 43 | all 3 by **bare-Kernel rule** |
| az_res_v20 (first 6, from coordinator) | 6 | 3 (50%) | 3/3 | **0/3** | | 2 bare-Kernel, 1 Trace burn then Kernel capture |
| v6 + v20 | 17 | 11 | **8/9** | **3/8** | | Fisher one-sided p ~ 0.04 for the colour split |
| hand_v2 (for reference) | 13 | 10 | 5/6 | 5/7 | | |

**Wins (8):**
- By Kernel capture: IC 3 (10304, 10359, 10616), KE 2 (10792, 11154), DE 1 (10970), TR shot 1 (11267).
- **10667 (MAG, 94 actions):** won by the bare-Kernel rule. CYAN took our last piece at a94.
- No firewall burns.

**Losses (3):**

| match | colour | end | material path |
|---|---|---|---|
| 10773 | MAG | a40 KE F5->E6 xIC: we took their last piece at Chebyshev distance 2 from their Kernel, so we (the side that just moved) lost | a24 DE x DE, recaptured by their Kernel -> K+TR vs K+IC -> a38 IC shot our Kernel, the lone guard TR died -> bare K vs K+IC |
| 11057 | MAG | a64 KE x EX next to their Kernel (forced; -1.00 from a63) | a8 warmup sample (p=0.04, Q -0.13 vs +0.16) -> a10 PI lost; a16 DE hung (a18) -> a30 TR lost -> bare Kernel from a30, 34 more actions |
| 11362 | CYA | a27-28 their Kernel takes our last piece; bare Kernels, our Kernel on A1 (ring 0) vs theirs on ring 2, so the ring rule lost it | a17-a18 walked TR into a 2-action Exploit strike (a19-20) -> K+DE vs K+TR -> a26 DE shot their Kernel, the guard died, our DE was left next to their Kernel |

**Near-disasters in won games:**
- 10359: a6 warmup, root800 -0.56 (recovered).
- 11154: a18, -0.30.

## 3. The systematic weakness: low-material endgames as Magenta (all 6 v6+v20 losses)
**The pattern:**
- Every one of the 6 losses (v6 x3, v20 x3) is a **Kernel + at most 1 piece each** endgame.
- 5 of the 6 are MAGENTA games of the same shape:
  1. In rounds 8-9 our Kernel retreats to the G6/G7 corner of the shrinking board, with the **Trace as the only guard**.
  2. CYAN's last piece walks up and either shoots the Trace directly (12113 a38) or attacks the Kernel, so the guard dies (10773 a38, 11686 a38).
  3. That leaves a bare Kernel vs K+1. Capturing their last piece next to their Kernel then loses by bare-Kernel rule 1 (**10773, 11686 and 12113 all end at a40**).
  4. Or the Trace shuffles F7<->F6 on the ring that will lock, and burns at round 16 (11791 a64).
- The five games have **different openings** (no shared prefix, different opponents), so this is not one opponent replaying a line.

**The cause is the value, not the search budget.** The raw value is far too optimistic for Magenta in these positions. The 800-sim search finds the loss, but only after it is forced.

| position (side to move = us, MAG) | v6 raw / root800 | v20 raw / root800 | v4f raw / root800 |
|---|---|---|---|
| 10773 a35 (K+TR vs K+IC, r9) | +0.47 / -0.38 | +0.14 / -0.13 | +0.13 / -0.26 |
| 12113 a32 (r8, v20 game) | +0.36 / +0.27 | **+0.47 / -0.41** | +0.38 / -0.14 |
| 12113 a35 (r9) | +0.42 / -0.92 | +0.20 / -0.86 | +0.02 / -0.71 |
| 11791 a47 (K+TR vs K+DE, r12) | +0.35 / -0.23 | **+0.54 / -0.36** | +0.05 / -0.21 |
| 11791 a59 (r15) | -0.15 / -0.81 | **+0.31 / -0.80** | +0.07 / -0.75 |

**Calibration over all live low-material positions** (each side K + <=1 piece; 268 positions from 13 v6/v20 games, `v6_lowmat.py`). Mean raw value when the side to move went on to lose:

| side to move | v6 | v20 | v4f | n |
|---|---|---|---|---|
| CYAN | -0.86 | -0.91 | -0.83 | 38 |
| **MAGENTA** | **-0.26** | **-0.27** | **-0.20** | 93 |

The positions within a game are correlated, so the n overstates the evidence. Still, the gap is large and has the same sign in all three models.

**Systematic or noise?**
- The win/loss colour split alone is weak evidence: 8/9 vs 3/8, p ~ 0.04 on 17 games. hand_v2 (5/7) and stopgap (6/7) did fine as MAGENTA.
- The **mechanism** is identical in all 5 MAGENTA losses across two models, though, and the value bias is measurable.
- **Verdict: systematic.** It is a blind spot of the shared hand_v2 base plus residual, and v20 has it **at least as badly** (its largest errors are in 11791/12113).
- hand_v4f's g1_t2/hang2/bare features shrink the raw error (v near 0 instead of +0.4) but still do not call these positions lost.

## 4. Weaknesses, ranked by live damage (v6 losses; v20 and v4f re-checked at the same positions)

**W1. Low-material endgame value blind spot (losses 10773, 11362, and v20's 11686/12113/11791).**
- **Evidence:**
  - Section 3.
  - 11362 a25-a26 (CYAN): the DE shot their Kernel at prior 0.95, the guard died, our DE was left next to their Kernel, and the bare-Kernel ring rule lost it with our Kernel still on A1.
  - At a25 v20 picks A1->B1 and v4f picks A1->B2 at 800 sims (moving the Kernel off the corner, Q -0.26 / -0.10 vs -0.78 / -0.63 for the played F3->F6). At 100 sims both still pick F3->F5 (Q -0.76 / -0.56).
- **Still there:**
  - v20: **yes, worse**.
  - v4f: yes, smaller.
- **Fix (in priority order):**
  1. **Endgame-seeded self-play for the residual.** Start games from:
     - the live positions 10773 a31, 12113 a29, 11686 a31, 11791 a41, 11362 a21 (both colours to move);
     - random placements of K+{TR,DE,IC,EX} vs K+{one piece}, sometimes with one extra piece, on the open board of rounds 8-16.

     Play at 100-200 sims. Train the value on **z only** for these positions (Q is biased exactly here). These games are short, so a few thousand fit in minutes on 2 workers.
  2. **Hand feature in the base (v4f graph), gated on `lowmat = (n_own_nonK <= 1) * (n_en_nonK <= 1)`:**
     - `lowmat * (ring_depth(ownK) - ring_depth(enK))` on the *currently open* board, with the Magenta-tiebreak sign;
     - `lowmat * g1_own * en_reach2(ownK or own guard)` (the lone guard can be shot or forced off within 2 actions);
     - `lowmat * guard_on_next_lock_ring` (11791).

     Hand-set weights of about 0.4-0.6, then refit.
  3. Optional prior term: penalise Kernel moves *toward* the open-board corner when `lowmat`, and moves that trade into `lowmat` while our Kernel ring is shallower than theirs.

**W2. Warmup sampling (11057 a8, still present; also 10773 a3 and 10359 a6 in won games).**
- **Evidence:**
  - 6 warmup decisions lost more than 0.2 Q; mean Q-loss is 0.080 per warmup decision vs 0.018 after.
  - In 24/44 warmup decisions the move played was not the greedy-100 move.
  - 11057 a8: PI F5->E5 at p=0.04, Q800 -0.13 vs +0.16, and the Pinger fell at a10.
- **Still there:**
  - v20: yes. It puts p=0.05 on the move, Q -0.48, and its prior is just as flat (top-1 0.10).
  - v4f: flat too (top-1 0.08), but its greedy choice is F5->F4 (+0.09).
- **Fix:** the warmup-only logit sharpening (hand_v2 review W1; v4f's `warm=0.5`) does not help when the prior has no signal. Raise the prior's top-1 in rounds 1-2 by distilling 800-sim visit targets on opening positions. The residual policy head can learn this directly: train dp on MCTS-800 visits for actions 1-8.

**W3. 2-action strikes on non-Kernel pieces (11362 a17-a18, 11057 a16).**
- **Evidence:**
  - 11362 a18: TR A3->B3 walked into EX F5->D4->B3 (Q800 -0.20 vs +0.04; a17 also dQ 0.23).
  - 11057 a16: DE F8->F7, hung to the Ice (Q -0.56 vs -0.37).
- **Still there:**
  - v20: avoids 11362 (c100 C3->C5) but **still plays 11057 a16 F8->F7**.
  - v4f: avoids both (c100 A1->B2 and F8->E8). This is its hang2 feature working.
- **Fix:** move the residual onto the v4f base (or port v4f's hang2/reach2 features), as planned.

**W4. Capture-into-Kernel-recapture trades at high prior (10773 a24).**
- **Evidence:**
  - DE F3->C3 xDE at prior 0.79-0.88 in all three models, then their Kernel recaptured.
  - v6-800: retreat F3->F6 at +0.49 vs +0.14.
  - v20-800 and v4f-800 keep the trade (+0.22 / +0.19).
- **Still there:** inconclusive. It is a real error only if W1 is real: the trade leads into exactly the K+1 vs K+1 endgame that W1 misjudges.
- **Fix:** follows from W1. Also check that the guard-aware capture prior (v4f) discounts captures whose landing square the enemy Kernel can reach.

**W5. Minor: value saturation and search depth.**
- |v| > 0.9 at 20.9% of live positions (hand_v2: 7.5%). Raw v correlates with root800 at **0.94** (MAE 0.13), but only **0.35-0.60** in the lost games 11362 and 10773: the blind spot is exactly where it matters.
- Prior top-1 median **0.24** (hand_v2: 0.115); 800-sim choice = prior top-1 in 60% of decisions (hand_v2: 40%). The residual sharpened the prior usefully.

## 4b. Tactical-pattern check: does v20 misjudge them? (`v6_probe3.py` -> `v6_probe3.txt`)
All values and Q are for the side to move. "c100" is that model's own 100-sim choice. Positions come from the live losses.

| pattern | position | v20 raw v / root800 | v20 top prior | v4f raw v / root800 | v20 misjudges? |
|---|---|---|---|---|---|
| (a) shot on a defended piece or lone guard | 12113 a38 (CYAN, al1, DE E6 shoots TR F7) | **+0.03 / +0.97** | shot 0.42 (top) | +0.03 / +0.95 | **Yes (value)**; prior OK. v4f same. |
| (a) the same threat, 2 actions out | 12113 a37 (CYAN al2) | +0.33 / +0.93 | 2nd (0.15) | +0.17 / +0.90 | **Yes (value, -0.6)**; v4f same |
| (a)+(b) Kernel attack that kills the lone guard | 10773 a37 (CYAN al2) | +0.19 / **+0.12** | top 0.16 | +0.08 / **+0.50** | **Yes, even v20's 800-search** (v4f's search finds it) |
| (a)+(b) | 11686 a37 (CYAN al2, won by CYAN) | -0.36 / **-0.29** | top 0.32 | -0.16 / **+0.33** | **Yes, even at 800** (v4f's search finds it) |
| (b) Kernel strike from a recapturable square | 11362 a26 (CYAN al1, DE F6 shoots G7) | -0.81 / -0.51 | **strike 0.82** (Q800 -0.74 vs -0.46 F6->C6) | -0.12 / -0.23; strike 0.32 (Q -0.53 vs -0.07) | **Yes (prior)**: c100 plays the strike; v4f's c100 too, but v4f's c800 does not |
| (c) threat that exists only with 2 actions | 11362 a19 (MAG al2, EX F5->D4->B3 xTR) | **-0.29 / +0.39** | 2nd (0.10) | +0.19 / +0.26 | **Yes (value, -0.68)**; v4f fixed (hang2) |
| (c) | 11057 a17 (CYAN al2, IC strike on DE) | +0.42 / +0.67 | 2nd (0.18) | +0.11 / +0.29 | Mildly (-0.25); v4f similar |
| (d) last capture before bare Kernels | 10773 / 11686 / 12113 a40 (MAG al1, every move -1.00) | **+0.34 / +0.56 / +0.41** vs -1.00 | suicide capture 0.93-0.95 | -0.45 / -0.33 / -0.57 | **Yes, badly (wrong sign)**; v4f has the sign right (its `bare` features) |
| (d) one action earlier | 11686 a39 (MAG al2) | +0.25 / -1.00 | | -0.41 / -0.95 | **Yes**; v4f partly |

**Reading:**
- v20 misjudges **all four patterns**, mostly in the value: errors of 0.6-1.0, and the wrong sign for (d).
- v4f already fixes (c), roughly halves (b) in the prior, gets the sign of (d) right, and its 800-search resolves (a)+(b) where v20's does not.
- What v4f still lacks is the magnitude for (a) and (d). After a guard-killing shot, the K vs K+1 position is simply lost, and neither model knows it (12113 a38: +0.03 vs +0.97 for both).

**Decision:** a tactics bundle is worth building only as **v4f base + low-material/bare-Kernel terms**. Re-doing (c) is not needed: v4f has it.
- (d)/(a), value: `lowmat` (each side <= 1 non-Kernel piece) x {own minus enemy Kernel ring depth on the open board, Magenta-tiebreak sign; `n_nonK == 0` for one side with the enemy Kernel within reach -> strongly negative; lone guard hittable by an enemy shot within 2 actions}.
- (b), prior: discount Kernel strikes whose origin square is adjacent to the enemy Kernel (the shooter stays put and is recaptured).
- Plus the endgame-seeded self-play in W1. It addresses the same blind spot without any graph change.

## 5. 100-sim vs 800-sim disagreement (v6 on both sides, greedy, referee tie-break)

| decisions | 100 != 800 | median dQ | dQ > 0.1 | dQ > 0.2 | dQ > 0.3 | mean Q gain/decision |
|---|---|---|---|---|---|---|
| all 219 of ours | **62 (28.3%)** (hand_v2: 35.6%) | 0.035 | 14 | 6 (2.7%) | 3 | 0.019 (hand_v2: 0.030) |
| post-warmup 175 | 49 (28.0%) | 0.027 | 8 | 6 | 3 | 0.018 |
| warmup 44 | 13 (29.5%) | 0.087 | 6 | 0 | 0 | 0.024 |

Largest gaps:
- 10792 a9 (+0.35, won)
- 10773 a24 (+0.34, lost: W4)
- 11267 a19 (+0.32, won)
- 11362 a17/a18 (+0.23 each, lost: W3)
- 11057 a16 (+0.18, lost: W3)

The disagreement rate is lower than hand_v2's because the sharper prior concentrates the 100 sims. The caveat from the hand_v2 review applies more strongly here: the 800-sim reference shares the W1 value blind spot. In the endgame losses, 100 and 800 sims agree on the losing moves (12113 a35: both -0.93; 11791 a55-a60: both play F7->F6).

## 6. Bottom line
- v6 is valid and is the file that played live (175/175 reproduced). It scored **8/11 (73%)**, CYAN 5/6, MAGENTA 3/5, with no swap forfeits. The local validator fails only under machine load.
- **Every loss of v6 and of v20 so far is a Kernel + <=1 piece endgame decided by the lone guard and the bare-Kernel rule. 5/6 of them were as MAGENTA.** All three models (v6, v20, v4f) overrate Magenta there: mean raw v -0.2 to -0.27 in positions Magenta then lost, against -0.85 for Cyan.
- **v20 still has it, as badly or worse. v4f reduces it but does not fix it.**
- **Concrete fix for the next model, in order:**
  1. Endgame-seeded self-play with z-only value targets (W1). Highest value, and it needs no graph change.
  2. Build on the v4f base (fixes W3; partly W1).
  3. Add a `lowmat` x Kernel-ring-depth / lone-guard-reach / guard-on-lock-ring feature trio.
  4. Opening policy distillation from 800-sim visits (W2).
