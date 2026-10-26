# Endgame solving (retrograde analysis, tablebases, endgame-specific evaluation/training) applied to COLONY: ONE low-material endgames

Context check (read before acting): the brief says v20 is live, but `gate/RECOMMEND.md` (15:18 UTC, "UPLOAD: az_res4f_v2.onnx") and commit a97550c ("deploy az_res4f_v2 (65.5% vs az_res_v20)") say **az_res4f_v2 is the upload**. az_res4f_v2 already contains 4000 endgame-start games (26% of samples). Everything below applies to both models, and was measured on both.

Local measurements in this note come from two scratch scripts run on the team's own endgame-start data (`agents/az/data/games_end_train1.pkl` + `games_end_holdout1.pkl`: 4800 games, 23,282 positions, 100-sim self-play, mean length 4.85 actions). Scripts: `C:\Users\melis\AppData\Local\Temp\claude\C--Users-melis-Downloads-cloudflight-coding-contest\2537b5b1-8fcc-4ac8-8fd3-b627f7b7b210\scratchpad\eg_stats.py`, `eg_val.py`, `eg_val2.py` in the same folder. "emp" = 2*(side-to-move win rate) - 1 in that data. These are outcomes under imperfect 100-sim play, not exact game values.

## Q0 (objective): fastest way to make the evaluator correct in low-material endgames, ranked and time-boxed

### Takeaway
Before 16:20, the best option is **R1: an exact, gated "bare-Kernel 1-action" value override inside the ONNX graph**. It has zero effect outside its gate (parity-checkable), costs about 30-40 extra nodes, and targets exactly the leaf type that v20 and az_res4f_v2 still misjudge. R2 (targeted endgame seeds + exact-label rescoring + retrain) is the second choice and probably does not fit together with gating. A full tablebase (exact retrograde) is post-contest.

### Cited Findings
- On random endgame starts, the deployed nets are already **calibrated on average** for "bare Kernel vs K+1" and "K+1 vs bare Kernel" (holdout, 3840 positions):
  - bare vs K+1 (side to move is bare): v20 mean v -0.65, az_res4f_v2 -0.63, emp -0.62.
  - K+1 vs bare: v20 +0.95, az_res4f_v2 +0.97, emp +0.93.
  - [local eg_val.py on games_end_holdout1.pkl](file:///C:/Users/melis/AppData/Local/Temp/claude/C--Users-melis-Downloads-cloudflight-coding-contest/2537b5b1-8fcc-4ac8-8fd3-b627f7b7b210/scratchpad/eg_val.py)
- The error concentrates in one sub-configuration: **side to move is bare, has 1 action left, and its Kernel is adjacent to the enemy's last piece X**. [local eg_val2.py](file:///C:/Users/melis/AppData/Local/Temp/claude/C--Users-melis-Downloads-cloudflight-coding-contest/2537b5b1-8fcc-4ac8-8fd3-b627f7b7b210/scratchpad/eg_val2.py)
  - **X within Chebyshev 2 of the enemy Kernel** (the capture is suicide by rule 1):
    - v20: v = -0.52 (train, n=321) / -0.62 (holdout, n=59), positive in 17-20% of positions; emp -0.83 / -0.90.
    - az_res4f_v2 (holdout): v = -0.56, positive in 12%; emp -0.90.
  - **X ≥ 3 from the enemy Kernel** (the capture can win via the ring rule):
    - v20: v = +0.15 / +0.27; emp +0.45 / +0.79.
    - az_res4f_v2: +0.12; emp +0.79.
  - Bare with 1 action and **not adjacent** to X: well calibrated (v20 -0.80 vs emp -0.83).
- This is the live (d) pattern. At 10773/11686/12113 a40 (MAG, al1, every move -1.00), v20 raw is +0.34/+0.56/+0.41 and the suicide capture has prior 0.93-0.95. v4f gets the sign right (-0.45..-0.57) but not the magnitude. — [reviews/az_res_v6.md §4b](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/reviews/az_res_v6.md)
- Order at turn end: the referee resolves the bare-Kernels rule **before** the firewall lock. `_end_turn` docstring: "resolve the bare-Kernels endgame and firewall"; the shrink happens only `if acting_team == MAGENTA and shrink_due`. — [colony_one/env.py lines 188-211 (read only)](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/env.py)
- The ring is `int(max(|r-3.5|, |c-3.5|))`: the centre 2x2 is ring 0 and the edge is ring 3. Smaller ring wins; equal rings go to Magenta; distance <= 2 loses for the `acting_team`. — [colony_one/board.py lines 184-245 (read only)](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/board.py)
- Observation planes (canonical, side to move):
  - 0-5 own Ping/Daemon/Exploit/Trace/Ice/Kernel; 6-11 enemy.
  - 12 locked squares.
  - 13 = actions_left/2 (1.0, then 0.5).
  - 14 = firewall urgency (1.0 = a shrink fires at the end of this round).
  - 15 = Magenta to move.
  - Magenta's view is rotated 180°. Ring and Chebyshev distance are invariant under that rotation.
  - [docs/observation.md](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/docs/observation.md); [encoding.firewall_urgency](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/encoding.py)
- Graph headroom: az_res4f_v2 has 278 nodes / 321 initializers (caps 1024/512) and runs 1.27 ms per batch-8 call. — [agents/az/RESULTS.md](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/az/RESULTS.md)

### Inferences

**R1, ≤30-40 min (recommended): exact gated override `v = v_net*(1-gA-gB) + gA*VA + gB*VB`.** Add it as torch ops in the ResModel wrapper before export, on top of az_res4f_v2.

Inputs: O = sum of planes 0-4, K = plane 5, E = sum of planes 6-10, EK = plane 11, open = 1 - plane 12, `al1 = [plane13 < 0.75]`, `mag = plane15`, `urg1 = [plane14 > 0.99]`. Constants: A1 = 3x3 ones without the centre, A2 = 5x5 ones, R = ring plane `int(max(|r-3.5|,|c-3.5|))` (one 8x8 initializer).

**Gate A: bare, 1 action, the enemy has exactly 1 piece.**
- `gA = [sumO == 0] * [sumE == 1] * al1 * (1 - urg1)`.
- urg1 is excluded because a lock after this turn could burn X or a Kernel.
- `kx = conv(K, A1)`: the squares our Kernel can step to.
- `winK = sum(kx*EK) * [sum(conv(E, A1)*EK) == 0]`: the enemy Kernel is adjacent and unguarded. Capture it and win.
- `capX = sum(kx*E)`: X is adjacent to our Kernel.
- `nearX = sum(E*conv(EK, A2))`: X is within 2 of their Kernel, so capturing it is suicide.
- `rX = sum(E*R)`, `rK = sum(EK*R)`.
- `capWin = capX * (1 - nearX) * clamp(rK - rX + mag, 0, 1)`. When rX == rK, the tie goes to Magenta.
- `VA = +1 if winK or capWin else -0.95`.
- -0.95 is the empirical value of handing K+1 two actions against a bare Kernel: the side with the piece wins 96-100% (Q2).
- Attacking a *guarded* enemy Kernel kills the guard and leaves both Kernels adjacent at turn end, so it loses. No term is needed for it.

**Gate B: K vs K, 1 action left** (we just captured with our first action).
- `gB = [sumO == 0] * [sumE == 0] * al1`.
- `VB = +1` if the enemy Kernel is adjacent (capture).
- `VB = +1` if there is a step square s with `kx*open*(1 - conv(EK, A2))` and `R(s) < rK` (or `R(s) == rK` and mag).
- Otherwise `VB = -1`.

**Policy prior (optional, same gates):** add -8 to the logit of "Kernel captures X" when gA and nearX hold. That removes the 0.93-0.95 suicide prior.

**Verification, about 10 min:**
1. Step-0 parity: max|dv| = 0 and max|dp| = 0 on the 3120 parity positions and the live positions where gates are 0.
2. On every gate-A/B position in the two pkl files (≈1.7k + ≈1k), compare VA/VB to brute force via `env.legal_actions()` + step on a board copy. The +1 options must match exactly.
3. Re-probe 10773/11686/12113 a39/a40 with v6_probe3.py. Expect ≈ -0.95.
4. The usual gate: validator, caps, 200-game h2h vs the live seat.

Because the change is a no-op outside the gate, the h2h only needs to show "not worse". Expected live effect: small but on-target.
- With 100 sims, the search reaches a39/a40 leaves from a37-a38, so it stops walking into the suicide line one turn earlier.
- It does **not** fix the earlier K+TR vs K+X misjudgement (a29-a36). That is class 1v1, where sign accuracy is only 0.73-0.78 (Q2). That needs R2 or a real tablebase.

**R2, ≈40-60 min, so probably too late for 16:20 with gating: targeted endgame data + rescoring + retrain (the az_res4f_v2 recipe).**
- Fix the start distribution.
  - `sp_end.random_start` only produces rounds 9-24 with ring 0 already locked. It produces nothing on the full 8x8 board at rounds 6-8, and nothing at rounds 25-26.
  - The live collapse starts at r8: 12113 a32 is r8, where v20 says +0.47 vs root800 -0.41.
  - Add about 30% round 6-8 full-board starts.
  - Add seeds "K + adjacent lone guard vs K+X with X able to hit the guard or the Kernel within 2 actions".
  - Add the 5 live seeds (10773 a31, 12113 a29, 11686 a31, 11791 a41, 11362 a21, both colours) with ±1-square perturbations.
- Rescore labels: replace z by VA/VB on gate-A/B positions (the Lc0 rescorer idea, Q3). Keep z elsewhere.
- Oversample by |v_net - z| (KataGo surprise weighting, Q3).
- Generate at 200 sims to cut conversion noise: emp is -0.83..-0.90 where the true value is ≈ -1.

**R3, post-contest:** an exact K+X vs K table and a K+X vs K+Y table for rounds 17-26 by forward-time backward induction (Q1), then distil it or add a material-gated endgame head (Q3).

### Gaps
- Time per `env.step` was not measured (1-core constraint, and the env is pure Python), so the R2 generation time is an estimate.
- A capture-legality edge case is not handled: R1 assumes a Kernel can always capture an adjacent enemy non-Kernel piece and can always make some step. The brute-force check in the verification step is what catches mistakes here.
- Not checked whether the referee's MCTS treats mid-turn K vs K as non-terminal. The rules say "resolve at the end of the current turn", so it should, and the holdout data contains 168 such al1 positions with nonzero net values.

## Q1: Feasibility of exact retrograde analysis (state space, forward DP vs retrograde, numpy solver timing, how chess/checkers/shogi do it)

### Takeaway
The game graph is a **finite DAG layered by (round, side, actions_left)**, so exact solving is plain backward induction over at most 72 half-turn layers (rounds 9-26). It needs no fixpoint iteration and no un-move generator. The size is fine for K+X vs K (≈15M states for rounds 9-26, ≈1.9M after D4 symmetry). But it needs either a reimplemented vectorised move generator (correctness risk) or pure-Python env calls (too slow). K+X vs K+Y on the 6x6 phase is ≈1.1 billion states. **Not doable before 16:20.** And the class that is cheap to solve (K+X vs K) is already 96-99% decided and calibrated.

### Cited Findings
- Classic retrograde analysis:
  - Start from the terminal positions (mates).
  - Generate predecessors with an un-move generator ("it is illegal to start in check, but legal to un-move into check").
  - A predecessor of a loss is a win. A candidate becomes a loss "only if all its successors are in W".
  - Iterate until no new positions appear. Positions are indexed bijectively ("Gödel numbers") into bitmaps.
  - History: Ströhlein (1970) did the first implementation; Thompson (1986) gave the DTM version.
  - [Chess Programming Wiki: Retrograde Analysis](https://www.chessprogramming.org/Retrograde_Analysis)
- The same page cites Bellman (1965): "the entire state-space could be stored and that dynamic programming techniques could then be used". Bellman also sketched combining forward search, DP and heuristic evaluation. — [CPW: Retrograde Analysis](https://www.chessprogramming.org/Retrograde_Analysis)
- Syzygy stores WDL50 and DTZ50. The 50-move counter makes some wins "cursed" (won but drawn under the rule): WDL returns 1 instead of 2, and DTZ > 100. The WDL probe assumes the position was reached directly after a zeroing move. A clock dimension changes the value of otherwise identical positions. — [python-chess Syzygy docs](https://python-chess.readthedocs.io/en/latest/syzygy.html)
- Chinook's checkers proof rests on endgame databases built by retrograde analysis. The largest covers 39 trillion positions. The 10-piece work (2001-2003) produced a 13-trillion-position database compressed to 148 GB. Checkers was weakly solved (a draw) in 2007. — [Schaeffer et al., Science 2007 (PDF)](https://www.cse.sc.edu/~mgv/csce580sp17/gradPres/schaeffer_CheckersIsSolved_Science2007.pdf); [IEEE Spectrum](https://spectrum.ieee.org/checkers-solved)
- In our referee:
  - Locks happen after rounds 8/16/24/26.
  - The game always ends after round 26.
  - The observation exposes urgency (time to the next lock), not the round itself.
  - [rules.md](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/docs/rules.md); [observation.md](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/docs/observation.md)
- The docs say: "Use the environment for legal actions and masks; do not reimplement the rules." — [observation.md](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/docs/observation.md)

### Inferences
**State counts** (ordered placements × piece types; 4 half-turn layers per round = side × actions_left; before symmetry):

| class | 8x8 (r≤8), per layer | 6x6 (r9-16), per layer / all 32 layers | 4x4 (r17-24) per layer / 32 | 2x2 (r25-26) per layer / 8 |
|---|---|---|---|---|
| K vs K | 4,032 | 1,260 / 40k | 240 / 7.7k | 12 / 96 |
| K+X vs K (5 types × 2 owners) | 2.50M | 428,400 / **13.7M** | 33,600 / 1.08M | 240 / 1.9k |
| K+X vs K+Y (25 type pairs) | 381M | 35.3M / **1.13B** | 1.09M / 35M | 600 / 4.8k |

- D4 symmetry is valid: the rings are D4-symmetric and no piece has a forward direction. It divides the counts by ≈8.
- Colour is **not** a symmetry, because Magenta wins ties.

**Structure of a solver:**
- Time strictly increases, so the graph has no cycles. That is unlike chess, where cycles force the fixpoint and un-move machinery.
- Solve layer L (round r, side, al) from layer L+1 using **forward** move generation plus a table lookup:
  - al2 → the same side with al1;
  - al1 → the opponent with al2 in the same round (after Cyan), or the next round (after Magenta, applying the lock first if r ∈ {8, 16, 24, 26}).
- Terminal leaves:
  - Kernel captures;
  - bare-Kernel resolution at turn end (before the lock);
  - the lock burning a Kernel (both burn → Magenta);
  - round 26 end.
- This is Bellman-style finite-horizon DP. The round index plays the role of Syzygy's 50-move counter: it is part of the state.
- Values within a phase probably converge as time-to-lock grows. That is unverified, so store all layers.

**Timing:**
- A pure-Python env enumeration of 13.7M states × ~15 actions ≈ 2×10^8 steps. At even 10 µs per step (env.step is pure Python, so probably slower) that is ≥ 35 min. **Not feasible in 10-20 min on one core.**
- A vectorised numpy move generator would be fast: only 3 pieces, so blocking is simple with precomputed rays. But it reimplements the rules (Kernel bodyguard order, Trace blocking, Daemon adjacent-enemy blocking, Exploit jump over locks). That is not a 30-min job to make exact. Post-contest.
- **The only class that plausibly fits a 10-20 min single-core env-driven run is K+X vs K on the 4x4/2x2 phase** (≈1.1M states, ≈135k after D4). Live losses are r8-r16 (6x6), so it is low value.
- The cheap exact piece is the **1-action bare-Kernel layer** (R1 in Q0). It needs no table: its successors are terminal or ≈ -0.95.
- How shogi tsume solvers fit in: see Gaps.

### Gaps
- I found no source here on shogi tsume solvers (df-pn / proof-number search), so they are not cited. The idea: they solve "forced Kernel kill within N turns" by OR/AND proof search, which would suit labelling K+guard vs K+X positions. Treat this as an uncited pointer for post-contest work.
- The actual env.step cost and the branching factor were not measured. The ≥ 35 min is an inequality from an assumed ≥ 10 µs per step.

## Q2: Analytic rules for these endgames (race to centre / parity, "never make the last capture within 2", K+1 vs K)

### Takeaway
The rules give exact facts that are cheap to encode:
- The last capturer must finish its turn ≥ 3 away from the enemy Kernel and on a smaller ring (equal ring is enough for Magenta).
- In rounds 25-26 the last capture always loses (unless it takes the Kernel).
- In rounds 17-24, a Cyan last capture can never win by the ring rule; a Magenta one wins iff the Kernels end ≥ 3 apart.

Empirically, **K+1 vs a bare Kernel is won by the side with the piece 96-99% of the time, whoever moves**. The bare side only wins when it is adjacent to the enemy Kernel with 2 actions (guard sacrifice, then Kernel capture: 98%). In K+1 vs K+1, Kernel ring depth is the dominant signal.

### Cited Findings
- Rule 1 (Chebyshev ≤ 2 → `acting_team` loses) is checked before the ring compare. The board.py docstring gives the reason: "it could be hit in the next round anyway". — [board.py 184-245 (read only)](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/board.py)
- Kernel protection: an attack on a guarded Kernel kills the cheapest adjacent guard (Ping, Exploit, Trace, Daemon, Ice), and the attacker stays at its origin. With no adjacent guard, destroying the Kernel ends the game immediately. — [rules.md](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/docs/rules.md)
- **Empirical outcome table** (100-sim self-play from random Kernel + ≤ 1 piece starts; side-to-move win rate; n positions): [local eg_stats.py](file:///C:/Users/melis/AppData/Local/Temp/claude/C--Users-melis-Downloads-cloudflight-coding-contest/2537b5b1-8fcc-4ac8-8fd3-b627f7b7b210/scratchpad/eg_stats.py)
  - **K+1 vs bare:** CYAN to move 0.977 (n=3290), MAG 0.989 (n=2797). Every distance and al bucket is 0.958-1.000. Every piece type is 0.963 (TR) to 0.994 (IC).
  - **Bare vs K+1:** CYAN 0.144 (n=2750), MAG 0.190 (n=3279).
    - By Chebyshev distance and al: d1 al2 **0.981**; d1 al1 0.231; d2 al2 0.328; d2 al1 0.067; d3 0.06-0.08; d4 0.09-0.13; d5 0.04-0.18.
  - **K+1 vs K+1:** CYAN 0.585 (n=4987), MAG 0.663 (n=5186). From start positions (2 actions in hand), the side to move wins 0.774.
    - By own-minus-enemy Kernel ring depth (positive = our Kernel more central):
      - +2: CY 0.78 / MAG 0.89
      - +1: 0.70 / 0.81
      - 0: 0.60 / 0.74
      - -1: 0.43 / 0.48
      - -2: 0.37 / 0.41
    - By our piece type: DE 0.76 and IC 0.79 vs PI 0.55, EX 0.58, TR 0.55.
  - **K vs K mid-turn (1 action left):** distance 1 → 1.000 (n=455, capture). Distance 2-4: 0.40-0.70.
- v20's value "does not credit Kernel centralisation in lowmat". At 12278 a49 the centralising B3->C3 had prior 0.01 and raw v -0.82 vs root800 +0.68. — [reviews/az_res_v20.md §3](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/reviews/az_res_v20.md)
- Our last non-Kernel piece was a Trace in all 6 live losses. — [reviews/az_res_v20.md §2](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/reviews/az_res_v20.md)

### Inferences
- **Exact geometric facts** from the ring = `int(Chebyshev to (3.5, 3.5))` definition:
  - 2x2 phase (r25-26): every pair of squares is within distance 1, so any capture that leaves bare Kernels loses for the capturer. Only a direct capture of an unguarded Kernel wins.
  - 4x4 phase (r17-24): open rows/cols 2-5. Distance 3 is only possible between opposite edges, and both are ring 1, so the rings are equal → Magenta.
    - A Cyan last capture never wins by the ring rule there.
    - A Magenta last capture wins iff the Kernels end ≥ 3 apart. Magenta can step away with its second action if the capture came first.
  - 6x6 phase (r9-16): the capturer needs distance ≥ 3 at turn end and ring(own) < ring(enemy) (≤ for Magenta).
  - Corner and edge Kernels (ring 2 on 6x6) lose every equal-or-worse comparison as Cyan. That explains why the G6/G7-corner Kernel retreats in the live losses are fatal.
- **"Never make the last capture within Chebyshev 2 at the end of your turn"** is exact. With 2 actions you can capture first and then step 1 square away: this escapes only if the capture landed at distance exactly 2 and a step to 3 exists. Encode it as the gate-A/B override plus the prior veto (Q0 R1).
- **K+1 vs bare Kernel is effectively won for the piece side.** The nets already know this (+0.95 to +0.97). The practical consequence for K+guard vs K+X: **losing the lone guard ≈ losing the game** (≈ -0.7 to -0.9) unless your Kernel is then adjacent to theirs with 2 actions.
  - So the value needs the feature "the enemy can remove our lone guard within its next 2 actions". That is v4f's hang2/thr: the shot on the guard, or a Kernel attack that sacrifices it.
  - The live 12113 a37/a38 positions (+0.03..+0.33 raw vs +0.93..+0.97 truth) are exactly this.
- **Race to the centre:** with 1v1 material, ring depth difference is worth ~±0.2-0.25 win probability per ring. Magenta's tie advantage adds ~+0.05-0.14.
  - The hand_v4f `kdiff` / `Kdepth_diff_bare` feature exists, but v20's base (hand_v2fast) has no such feature.
  - An exact feature `lowmat * (ring_en - ring_own)` with the Magenta-tie sign is cheap to add to the value MLP inputs. Retraining is needed to weight it.
  - A hand weight of ≈0.4-0.5 in tanh space matches the table: +1 ring ≈ +0.2 probability ≈ +0.4 in value.
- Daemon and Ice are the strongest lone pieces in 1v1 (0.76/0.79), Ping/Trace/Exploit the weakest (~0.55). Keeping a DE/IC over the TR as the last survivor matters; trades that leave our TR vs their DE/IC are bad.

### Gaps
- All win rates are from imperfect 100-sim self-play by the team's model, not exact values. The true K+1 vs bare value may be exactly +1 in nearly all positions. Unverified without a solver.
- The positions within a game are correlated (n overstates the evidence). The number of distinct games is printed in eg_stats.py output: e.g. 1366 games for the CYAN K+1 vs bare row.
- No exact proof that K+1 vs a bare Kernel is always a forced win (for example, a Trace alone may fail to force it near locks). The empirical TR rate of 0.963 is the lowest.

## Q3: How engines inject tablebase knowledge into nets, and which is quickest for our residual net

### Takeaway
There are three standard routes:
1. **Rescore training-game labels with exact results** (the Lc0 rescorer).
2. **Seed and oversample** the positions the net gets wrong (KataGo-style surprise weighting).
3. **A material-gated expert**: Stockfish NNUE selects one of 8 "layer stacks" by piece count.

For us, the quickest is an even stronger form of (3): a **hard-coded exact expert for the tiny bare-Kernel classes** (R1), which needs no training. Next come rescoring (1) plus oversampling (2) inside the existing az_res4f recipe.

### Cited Findings
- The Lc0 training-data rescorer uses Syzygy tablebases to correct training games with tablebase results / perfect play. Its notebook says "Rescoring is used in the training of all the best nets for leela, and should always be used before running training for best results." — [Colab mirror (David Chanin) on science.ecosyste.ms](https://science.ecosyste.ms/projects/175905)
- Studies of Lc0 against endgame tablebases (University of Alberta, Haque/Müller) show that even strong nets make measurable endgame errors where exact tables exist. — [Haque & Müller, "On the Road to Perfection? Evaluating Leela Chess Zero Against Endgame Tablebases" (ACG 2021)](https://webdocs.cs.ualberta.ca/%7emmueller/ps/2021/2021_Haque_ACG_Road_to_Perfection.pdf)
- Stockfish-style NNUE picks a layer-stack bucket from material: `bucket = (piece_count - 1) / 4`, i.e. 8 buckets. nnue-pytorch's training module defaults to `num_ls_buckets=8` and takes a per-sample layer-stack index. — [network.cpp (Stockfish-derived) on Hugging Face](https://huggingface.co/spaces/Amogh1221/deepcastle-api/blob/main/src/nnue/network.cpp); [nnue-pytorch lightning_module.py](https://isip.piconepress.com/projects/ai_chess/resources/pro_player_models/nnue-pytorch/model/lightning_module.py)
- KataGo "Policy Surprise Weighting" oversamples positions by the KL divergence between search result and prior: "roughly half the frequency weight is uniform and half is proportional to surprise". — [KataGo Methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
- Team precedent: in an offline experiment, a value target of 0.5*z + 0.5*root-Q gave 70% vs stopgap ("Q-mixing + clean teacher data >> outcome-only"). For endgame starts, z only was used because Q is biased exactly there. — [agents/az/RESULTS.md](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/az/RESULTS.md); [agents/az/sp_end.py](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/az/sp_end.py)
- How our model is structured: value = tanh(v_h + dv), policy = p_h + dp, and the value MLP sees the base's S/G feature vectors. — [agents/az/reslib.py](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/az/reslib.py)

### Inferences
- **Quickest (no training): a hard gated expert in the graph (R1).** It is a "mixture of experts" where the endgame expert is exact for its gate. Because the gate is a material count, it matches NNUE's material buckets, but with a computed value instead of a learned stack.
- **Next quickest (one retrain): rescoring.** Replace z by the exact VA/VB on the gate-A/B samples of the existing pkl files: 2.7k train positions, zero generation cost. Optionally label class K+1 vs bare as +1 / -1 instead of the noisy z (the empirical rate is ≥ 0.96).
  - Retrain with the az_res4f_v2 recipe, and oversample samples with high |v - z|.
  - This teaches the net the gate-A/B pattern, and the *preceding* positions inherit better Q through the 0.7*z + 0.3*Q mix in later self-play.
  - On its own it is weaker than R1 at the gate itself, which is exact.
- **Post-contest: a learned endgame head.** Add `dv_end` to the value: a small MLP on [S/G features, ring diff, guard-hittable flags], multiplied by `lowmat = [n_own ≤ 1][n_en ≤ 1]`. Train it on tablebase or solver labels. It needs no change to the middlegame path (no forgetting).

### Gaps
- I did not find primary documentation of the Lc0 rescorer's exact algorithm (whether it rewrites only z, or also the policy via a DTZ "policy boost"). Only the Colab mirror describes it.
- The KataGo methods page does not describe its start-position or hint-position machinery for starting games from non-initial positions. No cited mix ratio for start-position seeding was found.

## Q4: Making the team's endgame-start self-play effective (mix ratio, forgetting, outcome vs solver labels)

### Takeaway
The team's current 26% endgame mix did **not** cause measurable forgetting: az_res4f_v2 improved both endgame calibration and full-game h2h. The remaining problems are **distribution mismatch** (no round-8 full-board starts, no live-like guard-threat seeds) and **label noise** at 100 sims. Fix them with targeted seeds, rescored exact labels for the solvable sub-classes, and surprise oversampling. Keep the endgame share around 20-30%.

### Cited Findings
- az_res4f_v2 data: all saved self-play plus endgame-start self-play (4000 random Kernel + ≤ 1 piece starts, round 9-24 boards, outcome-only value targets), replicated to 26% of samples. — [agents/az/RESULTS.md](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/az/RESULTS.md)
- Endgame calibration MSE (az_res4f_v1 → v2 is the same recipe without → with endgame data): MAG 0.392 → 0.340, CYAN 0.413 → 0.320. v20: MAG 0.393, CYAN 0.423. — [agents/az/RESULTS.md](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/az/RESULTS.md)
- Full-game strength did not drop. az_res4f_v2 h2h: 152/200 = 76% vs v20 and 163/200 = 82% vs v6 (az agent); the gate measured 144/220 = 65.5% vs v20. az_res4f_v1 without endgame data was 50%. — [agents/az/RESULTS.md](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/az/RESULTS.md); [gate/RECOMMEND.md](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/gate/RECOMMEND.md)
- `sp_end.random_start`:
  - 80% of starts are rounds 9-16 with ring 0 locked; 20% are rounds 17-24 with rings 0-1 locked.
  - Each side gets 0-1 extra piece with probability 0.8; the piece is placed adjacent to its own Kernel 60% of the time.
  - Value labels: `q = z.copy()`.
  - [agents/az/sp_end.py](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/az/sp_end.py)
- Endgame-start games average 4.85 positions (4800 games). They are very short and tactical, so the data is cheap to generate. — [local eg_stats.py](file:///C:/Users/melis/AppData/Local/Temp/claude/C--Users-melis-Downloads-cloudflight-coding-contest/2537b5b1-8fcc-4ac8-8fd3-b627f7b7b210/scratchpad/eg_stats.py)
- Live failure shape: the Kernel retreats to the G6/G7 corner in rounds 8-9 with the Trace as the only guard. The opponent's last piece kills the guard (a38), and the game ends a40 by bare-Kernel rule 1. The collapse begins at r8 on the full board (12113 a32: v20 +0.47 vs root800 -0.41). — [reviews/az_res_v6.md §3](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/reviews/az_res_v6.md)
- KataGo's surprise weighting gives a cited recipe for oversampling misjudged positions: half uniform, half proportional to surprise. — [KataGo Methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)

### Inferences
- **Mix ratio:** keep 20-30% endgame samples. 26% showed no forgetting. The residual design (fixed base, small zero-initialised dv/dp heads) also limits how far endgame data can move middlegame outputs.
  - If more endgame data is added, raise its *diversity* (new seeds), not its replication factor. Replication of 4000 games to 26% already risks overfitting those specific starts.
- **Distribution:** the random starts reproduce the *average* class value but not the live configurations.
  - The nets are calibrated on average for bare vs K+1 (-0.65 vs -0.62).
  - They are wrong on the "bare, al1, adjacent to their last piece" sub-configuration and on the live a37-a40 positions.
  - Add:
    1. Round 6-8 full-board starts (no locked ring; urgency near 1). The first lock then hits during the game, which is what live 11791/13365 (Trace burned at r16) need. Also round 15-16 starts with pieces on the about-to-lock ring.
    2. "K + adjacent lone guard vs K+X, X within 2 actions of hitting the guard or the Kernel" seeds.
    3. The live seeds with perturbation.
  - Make sure both colours are to move.
- **Labels:** outcome-only is right for endgames given biased Q, but at 100 sims the opponent sometimes fails to convert. Example: emp -0.83..-0.90 on positions whose exact value is almost surely -1 (bare al1, suicide-only capture).
  - Use exact rescoring where it is exact (gates A/B).
  - Use ±1 for K+1 vs bare positions where the piece side won (≥ 0.96).
  - Generate at 200 sims for the rest.
  - Later, labels from a solver (Q1, post-contest) remove the noise entirely. This is the Lc0 rescorer pattern.
- **Order of operations if a retrain is attempted before 16:20:**
  1. Ship R1 on az_res4f_v2 first. It is independent and can be gated alone.
  2. In parallel, generate 2000 seeded endgame games at 200 sims on 2 workers (the games are short).
  3. Rescore and retrain with the v2 recipe.
  4. Gate only if the h2h screen at 16:10 shows Wilson LB > 0.5. Otherwise it is post-contest.

### Gaps
- No direct measurement exists of middlegame forgetting at endgame shares above 26%.
- The generation time for 2000 seeded games at 200 sims was not measured.
- Whether R1's override changes MCTS choices in the live positions earlier than a39 (a35-a38) is untested. A 100- and 800-sim re-probe of v6_probe3.py positions with the R1 model would answer it in minutes.
