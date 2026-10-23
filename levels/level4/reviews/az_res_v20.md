# Review: az_res_v20 (live since 14:53:30 UTC, 2026-10-23), 20 games

Sample: `matches_az_res_v20_b.json`, played_at >= 14:53:30: 20 games. All 20 have a protocol: 11621/11686/11791 come from `_a`, the rest from `_b`. The `score` field agrees with the protocol winner in 20/20.
"aN" = protocol action N. Q800/root800 = after an 800-sim search (greedy, no noise, referee tie-break), from the side to move. "lowmat" = each side has a Kernel + at most 1 other piece.

Scripts and outputs in `levels/level4/reviews/`:
- `v20_common.py` (loader), `v20_stats.py`, `v20_verify.py` -> `v20_verify.txt`
- `v20_scan.py` -> `v20_scan_az_res_v20.{txt,json}` (all 445 of our decisions, 100 vs 800 sims) and `v20_scan_hand_v4f.*` (the 3 new losses)
- `v20_probe.py` -> `v20_probe.txt`, `v20_probe_extra.txt` (31 critical positions: v20, hand_v4f, az_res4f_v1, distill_v5hf)
- `v20_lowmat.py` -> `v20_lowmat.txt` (299 lowmat positions, 4 models); `v20_ring.py` -> `v20_ring.txt` (Kernel ring depth)

## 1. Verification: PASS
- **Live seat runs this file:** local v20 MCTS-100 reproduces **365/365** post-warmup live moves across all 20 games.
- **Graph:** 204 nodes, 258 initializers (caps 1024/512), opset 17, `observation [batch,16,8,8]` -> `policy [batch,4096]`, `value [batch]`. sha256 61531c6c...
- **Validator:** ACCEPTED on the first try (now; 1 thread). Dry run 118 ms, peak RSS 5 MiB.
- **Outputs (877 live positions):**
  - All finite. Value in [-0.9996, +1.0000]; |v| > 0.9 in 12.5% of positions (v6: 20.9%). Logits in [-8.0, 9.8].
  - Latency at 1 thread: **0.42 ms per call at batch 1, 1.82 ms at batch 8**; session creation 62 ms CPU. That is about 2x faster than v6.

## 2. Live stats (20 games): 14 W / 6 L, CYAN 9/10, MAGENTA 5/10

| # | match | col | opp | res | len | end |
|---|---|---|---|---|---|---|
| 1 | 11621 | C | ? | W | 70 | Kernel capture (KE) |
| 2 | 11686 | M | ? | **L** | 40 | bare-Kernel |
| 3 | 11791 | M | ? | **L** | 66 | TR burned r16, then Kernel captured |
| 4 | 11912 | C | TrueSentinels | W | 33 | Kernel capture (KE) |
| 5 | 11982 | C | ConfusedMatrix | W | 36 | bare-Kernel (theirs) |
| 6 | 12113 | M | ? | **L** | 40 | bare-Kernel |
| 7 | 12278 | M | BluePhi09 | **L** | 56 | bare-Kernel |
| 8 | 12287 | C | The Posers | W | 34 | TR shot |
| 9 | 12411 | C | The Posers | W | 58 | KE |
| 10 | 12570 | M | ? | W | 36 | PI |
| 11 | 12607 | C | StochasticParrots | W | 37 | TR shot |
| 12 | 12768 | M | ConfusedMatrix | W | 56 | KE |
| 13 | 12845 | C | Trei patrate | W | 70 | KE |
| 14 | 13030 | M | ? | W | 36 | PI |
| 15 | 13136 | M | Babidjons AI | **L** | 40 | bare-Kernel |
| 16 | 13153 | C | ? | W | 33 | DE shot |
| 17 | 13353 | M | StochasticParrots | W | 24 | DE shot |
| 18 | 13365 | C | ConfusedMatrix | **L** | 66 | TR burned r16, then bare-Kernel |
| 19 | 13577 | C | ? | W | 18 | EX |
| 20 | 13658 | M | ? | W | 28 | TR shot |

- **Ends:**
  - All 14 wins end in a Kernel capture, except 11982 (their bare-Kernel loss).
  - **All 6 losses end via the bare-Kernel rule or a firewall burn, from a K+1 vs K+1 endgame.** None is a tactical Kernel loss in the middlegame.
- **Length:** median 38 actions; wins 18-70, losses 40-66.
- **By phase** (`v20_lowmat.txt`):

  | | reached lowmat | no lowmat |
  |---|---|---|
  | **CYAN** | 5/6 | 4/4 |
  | **MAGENTA** | **1/6** | 4/4 |

- **Our last non-Kernel piece in the endgame is a Trace in all 6 losses** (5 of 6 lowmat wins too: the Trace is simply the usual survivor). The opponent's piece in our MAGENTA losses: DE ×3, IC ×1, PI ×1.
- **Warmup is not a loss driver:**
  - 11 warmup samples had p ≤ 0.07 and cost 0.15-0.41 Q.
  - Only 11791 a3/a4 (dQ 0.15) is in a lost game.
- **100 vs 800 sims:** 24% disagreement; mean gain 0.020 per decision; 10 decisions with dQ > 0.2.
  - The largest gaps are in the won game 12768 (+0.50 at a35), i.e. MAGENTA endgame manoeuvring.

## 3. New losses (games 7-20), classified

All 3 belong to the **known low-material blind spot family**, but they add two specifics:
1. **Kernel centralisation** decides these endgames, and v20's value ignores it.
2. 13365 is the **first CYAN** instance (v20), reached via a Kernel strike whose shooter the enemy Kernel recaptured in **2 steps**.

### 12278 (MAG vs BluePhi09, 56 actions): known W1 (lowmat) + W4 (trade into lowmat)
1. **The trade into lowmat.** a39-a40 DE C6->C4->B3 shoots TR; a41-42 PI recaptures DE. Result: K+TR vs K+PI at r11.
   - At a40 v20's c800 prefers C4->C5 (+0.78 vs +0.57 for the shot); c100 plays the shot.
   - hand_v4f and az_res4f_v1 make the same choice. Only distill_v5hf's c100 avoids it.
2. **v20 stays confident while the position slips.** v20 thought MAGENTA was winning (root800 +0.65 at a43) while:
   - our Kernel sat in the G7 corner (mean ring 1.29);
   - the TR wandered F6-F5-E5;
   - CYAN's Kernel walked B3->C3->D4 to the centre (a49-50).
3. **The collapse comes within one action, a47 -> a48:**
   - root800 goes from +0.57 to -0.55; raw v from +0.86 to +0.47.
   - At a47 c800 prefers F5->F6 (+0.62 vs +0.43 for the played F5->E5).
4. **Value error from CYAN's side.** At a49 (CYAN to move) v20's raw value is **-0.82 for CYAN** vs root800 **+0.68**, and the centralising move B3->C3 gets prior 0.01. The value does not credit Kernel centralisation in lowmat.
5. **The finish.** a53-54: PI E4->G4->E6 is a 2-action shot on the lone Trace. Then K x PI and bare Kernels at distance 2, with CYAN's Kernel on ring 3 in any case.

### 13136 (MAG vs Babidjons AI, 40 actions): known W1 + pattern (a), lone guard shot within 2 actions
1. **Earliest lowmat of all 20 games.** a19-20 IC G7->E5->C3 xTR; DE recaptures. Result: K+TR vs K+DE at r6.
   - v20 Q800 +0.05 for the trade. Only distill_v5hf rejects it (Q800 -0.56, c800 H6->H5).
2. **Kernel placement.** Our Kernel stayed on the edge (G8-G7-F6-G5-G4, mean ring 0.95); CYAN's Kernel went B2->C5 (mean 1.58).
3. **Already lost at a31:** root800 -0.32, while raw v was +0.45.
4. **a35 TR F5->F4** (prior top, 0.30): raw v **+0.22**, root800 **-0.89**. hand_v4f is the same (+0.26 / -0.86).
5. **The finish.** a37-38: DE D5->E5 shoots TR F4. a39-40: K x DE at Chebyshev distance 2, so rule 1 applies. Same end as 10773 / 11686 / 12113.

### 13365 (CYAN vs ConfusedMatrix, 66 actions): NEW variant, the same family as CYAN
1. **a21-22: PI C4->E4->G6 strikes the Kernel.**
   - Prior 0.93, Q800 +0.36. All 4 models' c800 play it (distill_v5hf's prior prefers A1->B2 at 0.88, but its search still strikes).
   - The guard TR F6 is sacrificed, but the PI stays on E4. The MAG Kernel recaptures it in **2 steps** (G6->F5->E4) and lands on the centre square.
   - The v6 review's proposed prior fix only discounts strikes from squares *adjacent* to the enemy Kernel, so it would **not** catch this.
2. **Slow decline.** K+TR vs K+EX from a25 (root800 +0.03). The root then declines steadily: -0.38 at a45, -0.76 at a57, -0.93 at a61.
   - For 28 actions (a33-a60) CYAN shuffles its Kernel C4<->B4 while MAG's Kernel holds E3/E4 (mean ring 1.60 vs 2.71).
   - The Trace sits on B3, the ring that locks at r16.
3. **The burn.** a61-62 TR B3->C3->B3 burns at r16. This is cosmetic: root800 was already -0.93.
4. **The finish.** Bare K vs K+EX; the a66 K attack sacrifices the EX guard, so rule 1 applies.
5. **What v20 sees.** Its raw v swings by 0.4-0.6 between K on C4 and K on B4, so it sees ring depth locally. But neither 100 nor 800 sims finds a plan to contest the centre.

### Common mechanism (6/6 losses)
- **Kernel ring depth** (0 = outer 8x8 ring, 3 = centre) during the lowmat phase, from `v20_ring.txt`:

  | group | our Kernel | their Kernel |
  |---|---|---|
  | losses (mean) | 1.32 | 1.80 |
  | lowmat wins (mean) | 1.95 | 1.63 |
  | 12768, our only MAG lowmat win | 2.44 | 1.24 |

  Our Kernel was shallower or equal in 5/6 losses (all but 11686).
- **The sequence:** the side whose Kernel is shallower loses its lone guard (a Trace) to a 2-action shot or the firewall. Bare-Kernel rule 1 (capturing next to their Kernel) or rule 2 (ring) finishes it.
- **The value does not see it:**
  - Correlation of raw v with root800 is 0.79 over all decisions, but only **0.26-0.63** in 5 of the 6 lost games (13365: 0.76).
  - Every MAGENTA loss has 2-11 of our decisions where raw v exceeds root800 by ≥ 0.8. Examples: 11686 a40 +0.56 vs -1.00; 13136 a35 +0.22 vs -0.89; 12278 a48 +0.47 vs -0.55.
  - 13365 has none: there the error is a slow drift, not a cliff.

## 4. MAGENTA weakness: real, but it is an endgame weakness, not a colour weakness
- **The colour split weakens over time:**
  - 20 games: 9/10 vs 5/10, Fisher one-sided p = 0.07.
  - Games 7-20 only: 6/7 vs 5/7 (p = 0.50).
  - v6 + v20 together: 14/16 vs 8/15 (p = 0.044).
- **Split by game phase:**
  - MAGENTA outside lowmat endgames: **4/4**.
  - MAGENTA in lowmat endgames: **1/6** (CYAN in lowmat: 5/6).
  - This is the systematic part. The mechanism is the same in all 5 MAGENTA losses and in 13365.
- **Calibration on the 299 lowmat positions (all 20 games):** v20's mean raw v for the side to move when it later lost was -0.22 (MAG) and -0.34 (CYAN); sign accuracy 0.71. For comparison, positions where the mover went on to win average about +0.36 to +0.39.
- **Verdict: systematic.** The weak spot is the K+1 vs K+1 endgame, in particular Kernel centralisation and safety of the lone guard. It hits MAGENTA most, probably because the Magenta tiebreak makes the static value too optimistic. It is not a general MAGENTA weakness.

## 5. hand_v4f and the replacement candidates on the same positions: no better starting point
31 critical positions (`v20_probe*.txt`) plus the lowmat calibration:

| | sign-acc (299 lowmat) | MSE | MAG-to-move, later lost: mean v | CYAN-to-move, later lost: mean v | h2h vs v20 (RESULTS.md) |
|---|---|---|---|---|---|
| az_res_v20 | 0.71 | 0.70 | -0.22 | -0.34 | - |
| hand_v4f | **0.55** | **0.93** | -0.13 | **+0.01** | - |
| az_res4f_v1 | 0.75 | 0.66 | -0.21 | -0.29 | 50% (100/200) |
| distill_v5hf | 0.73 | 0.70 | -0.22 | -0.09 | 52% (52/100) |

- **hand_v4f is flatter (v near 0) and its search is blinder in these endgames:**
  - 12278 a48: root800 **+0.44** vs v20 -0.55. At a51: -0.20 vs -0.79.
  - 13365 a45 / a57: it rates CYAN's lost shuffle at **+0.40 / +0.19** (v20: -0.38 / -0.76).
  - Its policy is somewhat better in two places:
    - 13365: it would move the Trace off B3 earlier (c100 B3->B4 / B3->C3).
    - 12278 a47: it would not play F5->E5 (c100 G7->F6).
  - Its value would not have saved any of the 3 games.
- **az_res4f_v1 tracks v20 within about 0.1 on every probe** (same recipe), including 12278 a48 (-0.64) and 13136 a35 (+0.27 / -0.89). Same blind spot, same strength.
- **distill_v5hf is the only model that declines both trades into lowmat** (12278 a40 c100 C4->C5; 13136 a19 c800 H6->H5). It does not fix the endgame values (13136 a35: -0.20 / -0.69).
- **Conclusion:** building on hand_v4f does not by itself fix the failure mode that cost all 6 losses. A candidate is a real improvement only if it moves the lowmat calibration numbers above.

## 6. Recommendation and the 3 most valuable fixes
**Keep v20 live** unless a replacement both:
- (a) beats v20 head-to-head with Wilson lower bound > 0.5; az_res4f_v1 (50%) and distill_v5hf (52%) do not;
- (b) improves `v20_lowmat.py` on these 299 positions: sign accuracy > 0.75 **and** mean v of the side to move in lost MAGENTA positions < -0.4.

v20 is valid, about 2x faster than v6, and 14/20 live.

**Fixes, in priority order:**
1. **Endgame-seeded training, value target = z only (or a deep teacher).** This is the largest lever because it addresses all 6 losses.
   - Seed positions:
     - 12278 a43, 13136 a22, 13365 a25, 11686 a36, 11791 a27, 12113 a26;
     - random K+TR vs K+{DE, PI, IC, EX, TR} on the open board at r6-r16, both colours to move, Kernels on varied rings.
   - `agents/distill/endgame.py` already generates sparse endgames with a deep teacher. Feed its labels into the residual's value head.
2. **Kernel ring-depth term under lowmat** (graph feature, or at least a training emphasis):
   - `lowmat * (ring(ownK) - ring(enK))` on the open board, Magenta-tiebreak signed;
   - plus "lone guard hittable by an enemy shot within 2 actions" (13136 a35, 12278 a53) and "guard on next lock ring" (13365, 11791).
   - Evidence: shallower Kernel in 5/6 losses; at 12278 a49 the value is -0.82 vs search +0.68 for the centralising side.
3. **Discourage trades into lowmat while our Kernel is shallower,** including Kernel strikes whose shooter the enemy Kernel can recapture within **2** steps (13365 a22, prior 0.93), not just from adjacent squares.
   - distill_v5hf's prior already leans this way (12278 a40, 13136 a19, 13365 a22 prior), so it is the most promising base for this fix.

Caveat: "truth" here means the game outcome plus 800-sim searches with the same models. They share the bias, so the real errors are, if anything, larger.
