# Review: az_res4f_v2 (live since 15:19:40 UTC, 2026-10-23), 16 games

Sample: `matches_az_res4f_v2_a.json`, played_at >= 15:19:40: 16 games, all with a protocol, `score` agrees with the protocol winner 16/16.
"aN" = protocol action N (seed index k = N-1 actions applied). Q800/root800 = 800-sim greedy search, no noise, from the side to move. lowmat = each side K + <= 1 piece.

Scripts and outputs in `levels/level4/reviews/`: `v2_common.py` (loader), `v2_stats.py`, `v2_verify.py` -> `v2_verify.txt`, `v2_scan.py` -> `v2_scan_az_res4f_v2.{txt,json}` (all 300 decisions, 100 vs 800 sims), `v2_lowmat.py` -> `v2_lowmat.txt`, `v2_warm.py` -> `v2_warm.txt` (referee-noise sampling odds), `v2_probe.py` -> `v2_probe.txt` (12 critical positions; v2 / v20 / the `models/az_res4f_v3.onnx` snapshot of 15:32 UTC).

## 1. Verification: PASS
- Local v2 MCTS-100 reproduces **236/236** post-warmup live moves in all 16 games.
- Graph 278 nodes / 321 initializers, opset 17, `[batch,16,8,8] -> policy [batch,4096], value [batch]`, sha256 bca01a26679d92b6. (`gate/az_res4f_v2_warm.onnx` is a different file; the live seat matches `models/az_res4f_v2.onnx`.)
- Validator ACCEPTED first try (104 ms dry run, peak RSS 5 MiB).
- Latency 1 thread: 0.31 ms/call batch 1, 1.46 ms batch 8 (faster than v20). Session 47 ms CPU.
- 596 live positions: all finite, v in [-0.9999, +1.0000], |v| > 0.9 in 23.7% (v20 12.5%), logits [-9.5, 12.3].

## 2. Live stats: 10 W / 6 L, CYAN 6/8, MAGENTA 4/8

| match | col | opp | res | len | end | decided where |
|---|---|---|---|---|---|---|
| 14956 | C | Babidjons AI | W | 38 | KE captured | |
| 14998 | M | The Posers | W | 28 | TR shot KE | |
| 15112 | C | ? | W | 18 | PI x KE | |
| 15150 | M | ConfusedMatrix | W | 36 | K x K | |
| 15267 | C | ? | W | 6 | EX x KE | |
| 15273 | M | ConfusedMatrix | **L** | 46 | DE shot KE | lowmat K+TR v K+DE (from +0.46) |
| 15627 | M | Trei patrate | W | 40 | K x K | |
| 15702 | C | CVCT | W | 22 | IC x KE | |
| 15778 | C | Chip&Chris | **L** | 36 | K x K | **middlegame IC blunders a18-a26 (NEW)** |
| 15780 | M | Chip&Chris | **L** | 44 | bare-Kernel | warmup a3/a4 |
| 16003 | C | Pregateste-ma | **L** | 40 | PI x KE | warmup a2 |
| 16051 | M | Pregateste-ma | W | 64 | firewall r16 burned their KE | |
| 16211 | M | ByteM3 Pro | **L** | 54 | IC x KE | opening deficit by a7 |
| 16302 | C | ByteM3 Pro | W | 42 | K x K | |
| 16438 | M | CVCT | **L** | 38 | DE shot KE | warmup a4 + lowmat K+TR v K+DE |
| 16469 | C | ByteM3 Pro | W | 44 | bare-Kernel (theirs) | |

- Length: wins 6-64, losses 36-54.
- Ends changed vs v20: 5/6 losses end by a **Kernel capture**, only 1 by the bare-Kernel rule (v20: 6/6 bare-Kernel/firewall).
- In 4/6 losses our last guard sat next to our Kernel and was removed by an attack on the Kernel (Kernel-protection sacrifice: 15273 a42, 15778 a32, 16003 a32) or by a capture landing next to it (16438 a37). Then the Kernel fell.
- 100 vs 800 sims: 23% disagreement, mean gain 0.022/decision (v20: 24%, 0.020). 10 decisions with dQ > 0.2; the two largest are both in loss 15778 (a21 +0.71, a18 +0.41).
- Lowmat by colour: MAGENTA 4/8 (v20 1/6), CYAN 2/3. MAG lowmat **K+TR vs K+DE: 0/2** (15273, 16438). Against EX, TR, IC or a bare K: 4/4.

## 3. Losses classified

### 15273 (MAG vs ConfusedMatrix, 46): known lowmat pattern, K+TR vs K+DE, thrown from a winning position
- Root800 is +0.43..+0.47 at a19-a27. K x EX at a27 leads to K+TR vs K+DE.
- Value cliff:
  - a35: v **+0.33**, root800 -0.13.
  - a36: v +0.23, root800 -0.52.
  - Our Kernel shuffles F7<->G7 on ring 1 with the TR on G6.
- c800 deviates twice:
  - a27: F6->F5 instead of K G7->F7, dQ +0.24.
  - a35: F7->F6 instead of F7->G7, dQ +0.35 (Q800 -0.07 vs -0.42).
- Finish: a42 DE F4->F7 attacks the Kernel and the TR guard on G6 is sacrificed; a45-46 DE walks next to the bare Kernel and shoots it.

### 16438 (MAG vs CVCT, 38): warmup blunder, then the same K+TR vs K+DE cliff
- a4 DE F8->D8 (prior 0.01, sampled under referee noise) hangs the DE to PI B4->B6->D8. Q800 -0.56 vs -0.04.
- The game recovers to root800 about -0.05 at a24-a28.
- Lowmat from a26. c800 prefers K H8->G8:
  - a27: dQ +0.15 vs played TR G6->F6.
  - a28: dQ +0.16 vs played K H8->G7.
- a31: v **+0.38**, root800 **-0.71**. That is the largest raw-vs-search gap in these games.
- Finish: a35-36 TR G6->G7->G6 leaves the TR on row 6 next to K F7. a37 DE D6->G6 takes the TR, a38 DE shoots the Kernel.

### 15778 (CYAN vs Chip&Chris, 36): NEW, middlegame piece-safety blunders missed at 100 sims
- Root800 +0.35 at a18 (CYAN better). Then the IC tours:
  - a18 D4->B6: prior 0.04, Q800 +0.02; c800 D4->D5 +0.43.
  - a21 B6->D6: Q800 **-0.44**; c800 B6->C6 +0.27, dQ **+0.71**.
  - a25 D5->E6: Q800 -0.45; c800 A1->B2 -0.31.
- a27-28: TR G5->G4, then a 2-action diagonal shot takes the IC on E6.
- a26: raw v **+0.29**, root800 **-0.78**. The value does not see the 2-action TR move+shot (range 2-3 diagonal) on a loose piece.
- v20 and the v3 snapshot have the same blind spot: at a21 Q800(played) is -0.41 and -0.60; v3's prior on D4->B6 is still 0.03.
- Finish: a32 IC attacks K C2 and the TR guard on B2 is sacrificed; a34-36 our K takes the IC on E4, within 2 steps of their Kernel, which takes it. The game was lost already.
- This is not the endgame pattern and not a Kernel strike. It is a tactical (hanging-piece) horizon problem in the middlegame.

### 15780 (MAG vs Chip&Chris, 44): warmup variance
- a3 EX E7->F5 (prior 0.02, Q800 +0.15 vs +0.62) and a4 DE F8->D8 (prior 0.01, Q800 -0.56 vs +0.13).
- Root800 -0.52 by a7. It never recovers (-0.65 to -0.99) and ends bare-Kernel.
- Odds of sampling these under the referee's noise (`v2_warm.txt`, 60 seeds): **2% and 0%**.

### 16003 (CYAN vs Pregateste-ma, 40): warmup variance plus a material horizon
- a2 EX F1->G3 (prior 0.05) hangs the EX to PI G5->E5->G3. Sampling odds 0/60.
- Q800 of the move is only -0.24, but root800 is -0.61 two actions later: the value under-prices the hanging EX. v20 and v3 rate it -0.41.
- The game stays lost (-0.6 to -1.0). The same guard sacrifice (a32 EX attacks K C3, TR B3 dies) leads to the finish.

### 16211 (MAG vs ByteM3 Pro, 54): opening deficit, no single decisive error
- Root800 -0.19 at a3. After CYAN a5-6 TR B3->B4->xEX E7 (range-3 diagonal) it is -0.47 at a7; Q800 at a4 had said -0.20.
- MAG is a piece down from a14. c100 equals c800 on most moves, dQ ≤ 0.17.
- The lowmat end, bare K vs K+IC, was lost already.

### Verdict
- The known endgame weakness is **reduced but not fixed**:
  - MAG lowmat 4/8 vs 1/6 for v20.
  - On the 148 lowmat positions of these 16 games: v2 sign-acc **0.83**, MSE **0.375**; v20 0.73 / 0.501.
  - What remains is concentrated in **MAG K+TR vs K+DE** (0/2, both with raw v +0.3..+0.4 where search says -0.1..-0.7).
- **NEW:**
  1. A middlegame hanging-piece horizon (15778).
  2. Warmup noise decided 3 games (15780, 16438 start, 16003). Each sampled move had ≤ 3% odds. The DE F8->D8 blunder occurred twice.
- No recapturable-Kernel-strike loss this time.
- The v3 snapshot (models/az_res4f_v3.onnx, 15:32) already fixes the endgame values at the cliff positions:

  | position | v2 raw v | v3 raw v |
  |---|---|---|
  | 15273 a35 | +0.33 | -0.18 |
  | 15273 a36 | +0.23 | -0.43 |
  | 16438 a31 | +0.38 | -0.16 |

  Its c100 at 15273 a27 plays F6->F5, which is v2's c800 move. Lowmat sign-acc 0.86, MSE 0.348.
  It does **not** fix 15778 a18/a21.

## 4. Fixes for az_res4f_v3, with seed positions (k = actions already applied)
1. **K+TR vs K+DE endgame seeds (MAGENTA to move), self-play from these with targets from deep search / outcome:**
   - 15273 k=26, 34, 35, 38, 39.
   - 16438 k=26, 27, 30, 34.
   - Augment by mirroring and by moving our K one ring inward/outward.
   - Feature to add: "enemy DE/IC/EX can attack our Kernel within 2 actions while our lone guard is adjacent". That guard-sacrifice threat cost the guard in 4/6 losses: 15273 a42, 15778 a32, 16003 a32, 16438 a37.
2. **Middlegame hanging-piece seeds (2-action enemy move+shot/capture on a loose piece):**
   - 15778 k=17, 20, 24, 25.
   - 16003 k=1 (EX F1->G3 hangs to PI 2-step diagonal).
   - 16211 k=3 (TR B3->B4->E7 range-3 shot on EX).
   - Target: Q800 or deeper, not v100. A cheap feature: "our non-Kernel pieces attackable by an enemy move+shoot/capture within 2 actions" (TR diagonal 2-3 after a 1-step move, PI 2-step diagonal, DE slide+adjacent shot).
3. **Opening priors (warmup robustness):**
   - 15780 k=2, 3 and 16438 k=3 (DE F8->D8 hangs to PI B4->B6->D8, twice).
   - Train the prior so these get ~0 and the safe moves are concentrated. Under referee noise the odds are bounded at ~1-3% per bad move, so this is a small gain; lowest priority.
