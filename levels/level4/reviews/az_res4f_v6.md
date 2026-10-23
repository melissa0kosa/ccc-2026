# Review: az_res4f_v6 (live since 15:57 UTC, 2026-10-23), 8 games

Sample: `matches_v6_a.json`, played_at >= 15:57:30. That is 8 games, all with a protocol, and `score` agrees with the protocol winner 8/8.
- "aN" = protocol action N (seed index k = N-1).
- Q400/root400 = 400-sim greedy search, no noise, from the side to move.
- lowmat = each side has K + <= 1 piece.
- Light review: 1 CPU worker, 400 sims instead of 800.

Scripts and outputs in `levels/level4/reviews/` (the older `v6_*.py` files belong to the az_res_v6 review and were not touched):
- `v6_4f_common.py` (loader)
- `v6_4f_stats.py` -> `v6_4f_stats.txt`
- `v6_4f_verify.py` -> `v6_4f_verify.txt`
- `v6_4f_scan.py` -> `v6_4f_scan.{txt,json}`: all 164 of our decisions at 100 sims; the 2 losses also at 400 sims.
- `v6_4f_probe.py` -> `v6_4f_probe.txt`: v2's critical positions re-scored by v6, warmup sampling odds, lowmat calibration.

## Verdict: PASS. Keep v6 live.
- Both losses are to the same opponent (achilary), one in each colour. Both were decided in the opening or early middlegame: root -0.7 by a7, and -0.4 by a10 then -0.94 by a17.
- Neither loss is the known endgame pattern.
- v2's endgame value cliff (K+TR vs K+DE) is fixed at the v2 probe positions.
- The middlegame hanging-piece horizon is still present at 100 sims.

## 1. Verification
- **Moves:** local v6 MCTS-100 (greedy, no noise) reproduces **132/132** post-warmup live moves (k >= 8) in all 8 games. In the warmup, 18/32 match; the rest are noise or temperature samples, as expected.
- **File:**
  - md5 b4acd69e5486 (matches the deployed file), sha256 2af4b513317b8d7f, 1,023,282 bytes.
  - 278 nodes / 321 initializers, opset 17, `[batch,16,8,8] -> policy [batch,4096], value [batch]`. That is the same graph shape as v2.
- **Outputs:** 330 live positions, all finite. v in [-1.0000, +1.0000], |v| > 0.9 in 22.4% (v2 23.7%), logits [-9.3, 11.1].
- **Latency at 1 thread** (the machine is loaded by 3 sprints):
  - Model calls: 0.73 ms/call at batch 1, 2.76 ms at batch 8. Session create takes 78 ms CPU. v2, with the same graph on a quiet machine, measured 0.31 / 1.46 ms.
  - Full MCTS-100 decision: about 0.08 s wall. The longest live game (52 actions, 26 of ours) needs about 2 s, far below the 15 s CPU per match cap.
- **Caps / validator:**
  - The local validator now says "too slow to check" (2 s CPU / 3 s wall). That is machine load: RESULTS.md records the same message as load-induced.
  - v6 was ACCEPTED at export (RESULTS.md), and the server has played it in 8 games.
  - Not re-run on a quiet machine.

## 2. Live stats: 6 W / 2 L, CYAN 3/4, MAGENTA 3/4

| match | col | opp | res | len | end | decided where |
|---|---|---|---|---|---|---|
| 19499 | M | ? | W | 32 | PI x KE | root +0.6 from a7 |
| 19684 | C | CapyCode | W | 50 | K x K | |
| 19847 | C | Chip&Chris | W | 48 | bare-Kernel (their K x PI a47) | recovered from root -0.76 at a21 |
| 19881 | M | CVCT | W | 34 | bare-Kernel (their K x TR a34) | |
| 19980 | M | Pregateste-ma | W | 38 | bare-Kernel (their K x PI a37) | |
| 20020 | C | Trei patrate | W | 52 | bare-Kernel (their K x TR a52) | |
| 20222 | M | achilary | **L** | 30 | DE shot KE | warmup a4: TR hangs to a TR move+shot |
| 20258 | C | achilary | **L** | 46 | bare-Kernel (our K x TR a46, within 2 of their K) | opening slide a1/a5, then a13-14 DE hangs to IC |

- **Length:** wins 32-52, losses 30-46.
- **Win ends:** 4 of 6 wins end by the bare-Kernel rule after the opponent's Kernel took our last piece.
- **Loss ends:** one by a Kernel shot (DE). The other by the rule: our K took their last TR and landed in reach of their K, so the acting side loses. That position was already -1.00 for every move.
- **Not seen:** the guard-sacrifice finish from the v2 review.
- **100 vs 400 sims, post-warmup, losses only:** 6/27 decisions differ. Only one has dQ > 0.2: 20258 a13, +0.22.
- **Lowmat:**
  - MAGENTA 2/2 (K+TR vs bare K; K+PI vs K+IC). CYAN 3/4; the loss was bare K vs K+TR, lost long before.
  - No MAG K+TR vs K+DE endgame occurred.
  - On 107 lowmat positions, v6 sign-acc **0.80**, MSE **0.507**; v2 on the same positions 0.77 / 0.600.

## 3. Losses classified

### 20222 (MAG vs achilary, 30): opening variance; the warmup sample hangs the TR to a 2-action move+shot
- CYAN opens PI B4->C4, TR A3->B3; the TR on B3 eyes the C3->F6 diagonal.
- **a3** (before our first move): raw v **-0.31**, root400 -0.15.
  - We played TR H6->G6: prior 0.07; c100 F8->F7, c400 G5->F5; Q400 -0.12 vs -0.09.
  - Sampling odds under referee noise: 12/60.
- **a4, the critical position:** we played TR G6->F6.
  - Prior 0.05; sampling odds **4/60 (7%)**.
  - **Q400(played) -0.36 vs Q400(c400 G5->F5) -0.11**, dQ +0.25. c100 also plays G5->F5.
- **a5-6:** CYAN TR B3->C3, then shoots F6 (range-3 diagonal).
- **a7-9:** we recapture with PI x TR on C3, and their IC takes the PI. We are a piece down: root400 -0.71 at a7.
- **After a7:** c100 = c400 on every decision up to a27; the position is lost (-0.86 to -0.97 from a19).
- **a27-28:** the K walk G7->H6->G5 lets the DE take the EX and shoot the Kernel. c400 differs (dQ +0.12 / +0.04), but everything is below -0.87.
- **Class:** opening variance. The sampled move is the v2 "2-action enemy move+shot on a loose piece" pattern, here in the warmup.

### 20258 (CYAN vs achilary, 46): opening slide, then a middlegame hanging piece (post-warmup, at c100)
- **a1:** DE C1->B1, prior **0.01**, sampling odds **0/60** (a very rare noise sample). Q400 -0.14 vs +0.15 for c400 A3->B3.
- **a5:** PI C4->E4, prior 0.34, odds 23/60.
  - **c100 also plays it**, so this is a search error, not noise.
  - Q400 -0.42 vs c400 B1->C1 -0.12, dQ **+0.30**.
- **a6-12:**
  - The PI trade on F5 (our PI x PI, their EX x PI) frees G5.
  - a11-12: their TR G6->G5, then shoots our EX on D2 (range-3 diagonal).
  - Root400 -0.43 at a10. No saving move is found: c400 -0.41.
- **a13, the critical post-warmup position:** DE C1->C3, c100 = played.
  - Q400 -0.65 vs c400 B3->C3 -0.43, dQ **+0.22**.
  - a14 C3->C5 (Q400 -0.81 vs -0.73) walks the DE into IC G7->E7->C5 (move, then capture).
  - Raw v at a13/a14: -0.44 / -0.49. Root400 -0.52 / -0.76.
- **a17 onward:** root -0.94 after the trades; lost.
- **a30-46:** bare K vs K+TR at -0.99 to -1.00. a46 K x TR in reach of their K ends it by the bare-Kernel rule; every move was -1.00.
- **Class:** opening variance (a1, 0/60 odds), plus an opening search error (a5), plus a **middlegame hanging piece** (a13-14, the same 2-action capture horizon as v2's 15778).

## 4. v2's weaknesses: fixed or still present?
Source: `v6_4f_probe.txt`, with v2's critical positions re-scored by v6.

| v2 weakness | position | v2 | v6 | status |
|---|---|---|---|---|
| Endgame K+TR vs K+DE value cliff | 15273 a35 | v +0.33 | v **-0.17** (root400 -0.64) | **fixed** (value) |
| | 15273 a36 | v +0.23 | v **-0.11** | fixed |
| | 16438 a31 | v +0.38 | v **-0.75** (root400 -0.77) | fixed |
| | 15273 a27 | c100 K G7->F7 (v2 Q400 +0.27 vs +0.58 for F6->F5) | c100 **F6->F5** (Q400 +0.37, vs -0.37 for G7->F7; v2's c800 move) | fixed |
| Middlegame hanging piece (15778) | a18 | c100 D4->B6 | c100 **D4->D5** (= c400) | fixed here |
| | a21 | c100 B6->D6 | c100 B6->B4, Q400 -0.44 vs c400 B6->B5 +0.13 (dQ +0.57) | **still present** |
| | a26 | v +0.29, root -0.65 | v **-0.54** (sees the loss) | fixed (value) |
| Opening hangs (16003 a2, 16211 a4) | | c100 off c800 | c100 = c400 (B4->C4, F5->F4) | fixed at 100 sims |
| Warmup DE F8->D8 (15780, 16438) | | prior 0.01 | prior 0.01; not repeated live | unchanged (noise-bounded) |

- **Endgame:** the K+TR vs K+DE weakness is fixed in the values and in v6's 100-sim choices at v2's positions. It is untested live: no such endgame occurred in these 8 games.
- **Hanging pieces:** still present, now visible live in 20258 a13-14. v2's 15778 a21 is still missed at 100 sims (dQ +0.57).
- **Warmup variance:** still costs games (20222 a4 at 7% odds; 20258 a1 at 0/60), but with different moves than v2.

## What to fix next
1. 2-action hanging-piece horizon: train policy toward Q400+ targets on seeds 20258 k=12,13; 15778 k=20; 20222 k=3.
2. Opening vs achilary-style lines:
   - MAG after PI B4->C4 / TR A3->B3: avoid TR G6->F6 into the C3 diagonal (20222 k=2,3).
   - CYAN: avoid PI C4->E4 trades that free G5 for the TR->D2 shot (20258 k=0,4).
3. K+TR vs K+DE looks fixed; spend no more there. Keep v6 live unless a challenger clearly beats it. v8 at 58% h2h (Wilson LB 0.51) is borderline.
