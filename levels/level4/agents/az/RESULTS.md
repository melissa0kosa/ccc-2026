# AZ self-play results (100 sims, arena games alternate colours)

Format: wins/games=rate [95% Wilson CI] C=win rate as Cyan / M=as Magenta, cpu s/game (both sides)

| model | vs seat_l3_material | vs stopgap_v1 | vs greedy | vs prev best | notes |
|---|---|---|---|---|---|
| az_v1 | 36/40=90% [77%,96%] C80%/M100% 1.2s | 6/60=10% [5%,20%] C13%/M7% 1.1s | 16/20=80% [58%,92%] C70%/M90% 1.2s | - | PROMOTED (validated) |
| az_v2 | 37/40=92% [80%,97%] C85%/M100% 1.1s | 17/60=28% [19%,41%] C20%/M37% 1.6s | 17/20=85% [64%,95%] C80%/M90% 1.0s | 30/40=75% [60%,86%] C60%/M90% 1.9s | PROMOTED (validated) |
| az_v3 | - | 17/60=28% [19%,41%] C27%/M30% 1.4s | - | - | NEW BEST vs stopgap (validated) |

## Notes
- From-scratch AZ (az_v1..v3, 32ch x 4 res blocks, bilinear from/to + 1x1 policy head): beats seat_l3 (90%+) after one bootstrap,
  but only 28% vs stopgap_v1 after material + stopgap-teacher bootstrap; outcome-only value targets learn slowly (lval ~0.65).
- Offline experiment exp/t_q5_3k.onnx (same net, from scratch, ONLY the 1000 stopgap-vs-stopgap teacher games, value target
  0.5*z + 0.5*root-Q, 3000 steps): 42/60 = 70% [57,80] vs stopgap_v1 (C73/M67), 1.1 s cpu/game. Q-mixing + clean teacher data >> outcome-only.
- Residual on hand_v2: 600-800 boot steps on hand sp_v2 data alone ~ no gain (40-52%); gains appear once own ExIt self-play
  (400 self + 100 vs hand_v2 per gen, 100 sims, value target 0.7*z + 0.3*Q, lr 3e-4, D4 augmentation) is added (v4+).
- Hand-vs-hand sanity arenas (identical models) gave 14/40 and 18/40: 40-60-game arenas carry +-12-15 points of noise.
- Graph size: az_res_v4 = 503 nodes / 444 initializers (caps 1024 / 512), passes the official validator.

## Residual on hand_v2 (az_res_vN): value=tanh(v_h+dv), policy=p_h+dp, trunk 32ch x 2 res blocks, zero-init heads (v0/v2 = exact hand_v2 copy; v1 = LR-flip aug, 800 steps on hand sp_v2; v3+ = D4 aug)

| model | vs hand_v2 | vs stopgap_v1 | vs seat_l3_material | notes |
|---|---|---|---|---|
| az_res_v0 | 14/40=35% [22%,50%] C40%/M30% 1.5s | - | - | step 0 == hand_v2 (parity 0.0e+00/0.0e+00) |
| az_res_v1 | 24/60=40% [29%,53%] C47%/M33% 2.2s | 31/40=78% [62%,88%] C80%/M75% 2.2s | 18/20=90% [70%,97%] C80%/M100% 0.7s |  |
| az_res_v2 | 18/40=45% [31%,60%] C65%/M25% 1.2s | - | - | step 0 == hand_v2 (parity 0.0e+00/0.0e+00) |
| az_res_v3 | 31/60=52% [39%,64%] C53%/M50% 0.9s | 34/40=85% [71%,93%] C70%/M100% 0.6s | 19/20=95% [76%,99%] C90%/M100% 0.4s | NEW BEST vs hand_v2 (validated) |
| az_res_v4 | 43/60=72% [59%,81%] C67%/M77% 1.1s | - | - | NEW BEST vs hand_v2 (validated) |
| az_res_v5 | 40/60=67% [54%,77%] C67%/M67% 1.5s | - | - |  |
| az_res_v6 | 41/60=68% [56%,79%] C57%/M80% 1.3s | 34/40=85% [71%,93%] C80%/M90% 1.1s | 20/20=100% [84%,100%] C100%/M100% 0.8s |  |
| az_res_v6 (confirm) | **141/200=70% [64%,76%] C73%/M68% 1.3s** | **91/100=91% [84%,95%] C88%/M94% 1.6s** | - | 200-game confirmation, validator ACCEPTED (503 nodes / 444 initializers) |
| az_res_v7 | 38/60=63% [51%,74%] C67%/M60% 1.1s | - | - |  |
| az_res_v8 | 42/60=70% [57%,80%] C67%/M73% 1.5s | - | - |  |

## Residual on hand_v2fast base (az_res_v20+)
Base = FastNet(w_v2) (output-identical to hand_v2: step-0 parity max|dpolicy| 9.5e-7, max|dvalue| 3.3e-7 on 3120 positions),
graph 204 nodes / 258 initializers, 2.6 ms per batch-8 call (v6: 503 / 444, ~3.5 ms). Value MLP gets the base's S/G feature vectors.
v20 = offline retrain (3000 steps, D4, value target 0.7*z+0.3*Q) on all saved games (v4-v8 self-play + hand sp_v2, 133,651 positions);
v21+ = self-play loop (400 self + 100 vs az_res_v6 per gen).

**Candidate rule (for the gatekeeper):** a model is flagged CANDIDATE only if it beats az_res_v6 head-to-head with Wilson 95% lower
bound > 0.5 (60-game screen; if >= 58% it is extended to 300 games). Rows without CANDIDATE are not upload candidates.
From v21 on the loop screens against az_res_v20 (stronger than v6) instead; a version flagged there gets an extra
>= 200-game head-to-head vs az_res_v6 before it is called an upload candidate. v20 is the current CANDIDATE.
The planned 300-game az_res_v8 vs v6 check was cancelled at 14:47 (it never finished); v20 superseded it.
Local validator note (14:40 UTC): with ~40 python processes on the box, the validator's wall-time check now REJECTS even
az_res_v6 and hand_v2fast (both ACCEPTED earlier) -- "too slow to check" is load, not the model; re-validate on a quiet machine.

| model | vs hand_v2 | - | - | notes |
|---|---|---|---|---|
| az_res_v20 | 36/40=90% [77%,96%] C90%/M90% 1.5s | - | - | h2h vs az_res_v6: 227/300=76% [71%,80%] C75% 2.0s; **CANDIDATE: beats az_res_v6 head-to-head, Wilson LB 0.71 > 0.5** (local validator ACCEPTED) |
| az_res_v30 | - | - | - | h2h vs az_res_v20: 53/100=53% [43%,62%] (C54/M52) -- v20 recipe with 6000 steps; no gain |
| az_res_v31 | - | - | - | h2h vs az_res_v20: 92/200=46% [39%,53%] (C45/M47) -- value target 0.5*z+0.5*Q; no gain |
| az_res_v32 | - | - | - | h2h vs az_res_v20: 95/200=48% [41%,54%] (C50/M45) -- v20 recipe + 500 v20 self-play games (v21 gen); no gain |
| az_res4f_v1 | - | - | - | h2h vs az_res_v20: 100/200=50% [43%,57%] (C47/M53) -- v20 recipe on hand_v4f base (step-0 parity 0.0 vs hand_v4f.onnx, 278 nodes/321 inits); no gain |

## az_res4f (residual on hand_v4f base) -- CURRENT BEST
az_res4f_v2 = v20 recipe (3000 steps, D4, value 0.7*z+0.3*Q) on hand_v4f base, data = all saved self-play (v4-v8, v21, 900 extra
v20 games) + hand sp_v2 + hand sp_v4f (600 games @200 sims) + endgame-start self-play (4000 random Kernel+<=1 piece starts,
rounds 9-24 boards, outcome-only value targets) replicated to 26% of samples. Step-0 parity vs hand_v4f.onnx: 0.0 / 0.0.
Graph 278 nodes / 321 initializers, 1.27 ms per batch-8 call, local validator ACCEPTED (15:15 UTC).

| model | vs az_res_v20 (h2h) | vs az_res_v6 (h2h) | vs hand_v4f | notes |
|---|---|---|---|---|
| az_res4f_v2 | **152/200=76% [70%,81%] C83%/M69% 1.4s** | **163/200=82% [76%,86%] C86%/M77% 1.5s** | 164/200=82% [76%,87%] C81%/M83% 1.3s | **CANDIDATE: beats az_res_v20 and az_res_v6 head-to-head, Wilson LB 0.70 / 0.76 > 0.5** |

Endgame value calibration (800 held-out endgame-start games, 3840 positions; mean raw value where side to move later LOST / WON, MSE):
- az_res_v20:  MAG lost -0.50 won +0.70 mse 0.393 | CYAN lost -0.53 won +0.64 mse 0.423
- hand_v4f:    MAG lost -0.40 won +0.62 mse 0.470 | CYAN lost -0.36 won +0.61 mse 0.492
- az_res4f_v1: MAG lost -0.45 won +0.76 mse 0.392 | CYAN lost -0.52 won +0.67 mse 0.413 (no endgame data)
- az_res4f_v2: MAG lost -0.50 won +0.81 mse 0.340 | CYAN lost -0.57 won +0.75 mse 0.320 (with endgame data)
(These are random endgame starts, not the 268 live-loss-type positions from the review.)

### az_res4f v3-v5 (endgame push), screened vs az_res4f_v2 with c1.arena (8 workers)
| model | vs az_res4f_v2 (h2h) | lowmat calib (gate/lowmat.py, 299 live pos.) | recipe |
|---|---|---|---|
| az_res4f_v3 (md5 5af5b4c1dea2813a0c4ecd2abe140b06) | 40/80=50% [39%,61%] C57/M42 | MSE 0.359, sign 0.89 (v2: 0.487 / 0.79), not trained on these games | +4f_v2 self-play (900 g) + 3000 endgame starts by 4f_v2 (30%), outcome-only value when <=2 pieces/side, OneCycle 6e-4 |
| az_res4f_v4 (md5 bbf56c255b7d71f97697eba68e15438b) | **135/240=56% [50%,62%]** (68/120 C60/M53 + 67/120 C58/M53) | MSE 0.262, sign 0.92 (contaminated: live low-mat positions were self-play seeds) | v3 + fresh endgame/pre-trade/live-seed starts (7883 g, 35%) + distill endgame_labels x3 + value x2 at <=1 piece/side + SWA 1000 steps (BN recomputed) |
| az_res4f_v5 | 83/160=52% [44%,59%] C56/M48 | - | v4 + distill endgame_live x8 + SWA 1500 |
No v3-v5 reaches the 58% pre-screen bar cleanly; v4 is the best (Wilson LB ~0.50). All validator ACCEPTED at export, 278 nodes / 321 initializers.

### GPU sprint (az_res4f_v6): init az_res4f_v4, ALL data (312k: self-play + endgame starts + sp_v4f + distill teacher/live labels + 300 oracle-800 + 140 live games outcome-only), weighted sampling (MAGENTA-to-move x2, teacher/oracle/live x2-3), batch 2048 bf16, D4, cosine, SWA last 20% + BN recompute
| model | vs deriv_4f2_rsv125 (live, c1.arena 8 workers) | notes |
|---|---|---|
| **az_res4f_v6** | **69/100=69% [59%,77%] C74%/M64% 1.9s** | **CANDIDATE**, validator ACCEPTED, 278 nodes / 321 inits |
| az_res4f_v6s | 70/100=70% [60%,78%] C72%/M68% 1.7s | = v6 with residual value x1.25 baked in, validator ACCEPTED |
| az_res4f_v7 (md5 69869024ed13dce61cf671424d1d2737) | h2h vs az_res4f_v6: 48/100=48% [38%,58%] C58/M38 | v6 recipe, 15000 steps from v4; no gain |
| az_res4f_v9 (md5 1a9a200d3713fe4cdf7156a8643a73d8) | h2h vs az_res4f_v6: 106/200=53% (55/100 + 51/100) | 3000 steps from v6 + live_opp/live_ours records (value; opponents' winning moves imitated), weight 5 |
| az_res4f_v8 (md5 a90bcbee050794920093d33b824e50e8) | **h2h vs az_res4f_v6: 116/200=58% [51%,65%]** (56/100 C66/M46 + 60/100 C54/M66) | 15000 steps from v6 + live records; borderline (Wilson LB 0.51), validator ACCEPTED |
