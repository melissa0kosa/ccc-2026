# Distillation agent: results

Teacher: `teacher.py`, an alpha-beta negamax over single actions (2 per turn, side switch, Kernel capture, bare-Kernel endgame and firewall shrink all handled through the engine Board API). Search depth = rest of own turn + full opponent turn (3-4 actions), followed by a captures-only quiescence search over the side to move's remaining actions, with a transposition table. Static eval: material, firewall burn penalties (Kernel and pieces on the next ring to lock, scaled by rounds left), Kernel guards, centrality.
Labels: policy = softmax(root move scores / 0.5) over legal moves; value = 0.75*tanh(best/3) + 0.25*game result.
Student: `train.py`, arch `sg` = 32ch x 4 residual blocks plus a residual on top of the fixed-weight stopgap evaluator (stopgap features as extra input planes, stopgap prior and value as an additive base). Policy head = bilinear from/to (MatMul) + 1x1 conv.

All arenas: 100 sims, alternating colours, `c1.arena`.

| model | data | vs stopgap_v1 | vs seat_l3_material | vs greedy | vs prev best | cpu/game (both) |
|---|---|---|---|---|---|---|
| distill_tmpsg (r1 only, 1.8k pos) | r1 | 23/40 = 57% [42,71], cyan 45% / mag 70% | | | | 2.0 s |
| **distill_v1** (r1-r3 partial, ~3.0k pos, 40 ep) | r1,r2,r3 | 40/60 = 67% [54,77] (cyan 60 / mag 73); rerun 40/60 = 67% [54,77] (cyan 60 / mag 73); pooled 80/120 = 67% | 58/60 = 97% [89,99] (cyan 93 / mag 100) | 60/60 = 100% [94,100] | (first) | 1.2-2.3 s |
| distill_v2n (r1-r5, 6.7k pos, no child samples, 30 ep) | r1-r5 | 52/60 = 87% [76,93] (cyan 80 / mag 93) | | | vs v1: 40/60 = 67% [54,77] | 0.9 s |
| **distill_v2** (r1-r5, 6.4k pos + 14.8k child value samples, 30 ep) | r1-r5 | 51/60 = 85% [74,92] (cyan 83 / mag 87) | 59/60 = 98% [91,100] | 60/60 = 100% | vs v1: 51/60 = 85% [74,92]; vs v2n: 38/60 = 63% [51,74] | 0.6-1.7 s |

Teacher itself (alpha-beta, direct play, no MCTS) vs stopgap_v1 under the referee search: 19/20 = 95% [76,99], as cyan 9/10, as magenta 10/10. So the student still has a lot of headroom.

Data tags: r1-r3 = teacher-behaviour games (old eval), r4 = MCTS(distill_v1 / stopgap_v1) behaviour (DAgger), r5 = teacher behaviour; r4+ include "child" value samples (positions after up to 8 random root actions, value = teacher score of that action), firewall penalty strengthened from r4 on.
| exp_t03 (T_POL 0.3, r1-r5 incl. later r4/r5 games, 7.8k pos) | r1-r5 | | | | vs v2: 36/60 = 60% [47,71] | 2.2 s |
| exp_t10 (T_POL 1.0, same data) | r1-r5 | | | | vs v2: 34/60 = 57% [44,68] | 1.9 s |
| **distill_v3** (T_POL 0.4, r1-r7 so far: 11.4k pos + 53k child samples, 30 ep, sg arch 32x4) | r1-r7 | (v2 was 87% pooled) | | | vs v2: 37/60 = 62% [49,73] (cyan 63 / mag 60) | 1.7 s |
| **distill_v3hb** (same data, residual on hand_v3 FastNet instead of stopgap; equals hand_v3 at init) | r1-r7 | | | | **vs hand_v3: 47/60 = 78% [66,87] (cyan 73 / mag 83)** | 6.4 ms/call b8 under load (v3 3.2 ms, hand_v3 1.3 ms) |

Notes: the policy temperature (0.3 / 0.5 / 1.0) barely matters; more data and the child value samples are what help. distill_v3hb passes the validator, but only on 1 of 3 tries on this loaded machine ("too slow to check" = 3 s wall-clock budget for the whole child process; dry run 155 ms vs 100 ms for hand_v3).

Running at 14:40 UTC: r6 gen (MCTS distill_v2 / stopgap_v1 behaviour, DAgger) and r7 gen (teacher behaviour), 2 workers each until about 15:03, both with child samples; arena distill_v3hb vs distill_v3 and vs hand_v2 (log `arena_v3hb.log`); training distill_v4s8 (sg arch, all 8 board symmetries; the legal-move equivariance self-check `symcheck.py` gives 0 mismatches), log `train_v4s8.log`.

## Round 2 (14:40-15:35 UTC)

Gatekeeper (vs hand_v2): distill_v3 146/190 = 76.8% [70,82] (cyan 74 / mag 80); distill_v2 64.5%; distill_v2n 51%. **Strongest line = the stopgap-based residual (`sg` arch): distill_v3, then distill_v4s8.**

| model | data / recipe | result | dry run (validator, loaded machine) | nodes / inits |
|---|---|---|---|---|
| distill_v3hb vs distill_v3 | | 30/60 = 50% [38,62] (cyan 43 / mag 57) | 155 ms | 241 / 321 |
| **distill_v4s8** | sg 32x4, all 8 symmetries, r1-r7 (15.7k pos + ~100k child samples), 30 ep | **vs az_res_v6: 75/100 = 75% [66,82] (cyan 72 / mag 78) -> FLAG: beats the live seat, Wilson lower bound 0.66 > 0.5**; vs distill_v3hb: 51/100 = 51% [41,61] | 60-124 ms (in-process median 52 ms; hand_v3 68 ms) | 134 / 126 |
| distill_v4hb | hbs (hand_v3 base, no stopgap planes) 32x3, 8 symmetries, r1-r8 partial (17.7k + 103k), 15 ep | vs distill_v4s8: 47/100 = 47% [38,57] (cyan 52 / mag 42), so not better; the sg line stays best (and is faster) | 85-121 ms (in-process median 94 ms) | 211 / 264 |
| distill_v4s8, 200-game check | | vs distill_v3: 109/200 = 55% [48,61] (cyan 56 / mag 53); vs hand_v4f: 75/100 = 75% [66,82] (cyan 74 / mag 76) | | |
| distill_v5s8 | sg 32x4, 8 symmetries, r1-r8 (19k + 113k), 30 ep | vs distill_v4s8: 42/100 = 42% [33,52]: no gain from more data | (same graph as v4s8) | 134 / 126 |
| **distill_v5hf** | hfs = residual on hand_v4f FastNet4 (equal to hand_v4f at init), 32x3, 8 symmetries, r1-r8 (19k + 113k), 20 ep | **vs az_res_v20: 52/100 = 52% [42,62] (cyan 58 / mag 46)**: on par with the live seat, not clearly better | in-process median 183 ms (az_res_v20 132, v4s8 70); fails the local 3 s check under the current load | 285 / 327 |
| distill_v5hf vs distill_v4s8 | | 47/100 = 47% [38,57] (cyan 34 / mag 60): the two lines are about equal | | |

Endgame value calibration (`calib.py`, 400 random sparse endgames: Kernel + 0-2 pieces per side, rounds 6-25, labelled by a 2-extra-turn teacher search; target tanh(best/3)): MAE / sign agreement for hand_v4f 0.33 / 0.86, az_res_v20 0.31 / 0.86, az_res_v6 0.32 / 0.85, distill_v4s8 0.32 / 0.86 (magenta 0.35 / 0.82), distill_v5hf 0.29 / 0.87.
| distill_v6hf | v5hf fine-tuned (lr 6e-4, 8 ep) on r1-r9 (incl. r9 = DAgger games of v4s8 / az_res_v6 / v3hb / v3 with follow-up samples) + 4.3k random endgames (x3 weight), 8 symmetries | vs az_res_v20: 51/100 = 51% [41,61] (cyan 56 / mag 46) | ~183 ms | 285 / 327 |
| distill_v6s8 | v4s8 fine-tuned the same way | vs az_res_v20: 50/100 = 50% [40,60] (cyan 52 / mag 48) | ~52 ms | 134 / 126 |

No distill student is clearly ahead of az_res_v20 (all 47-52%). The plateau since v3 suggests the 32x4 student, not the data, is now the limit.

### Endgame labels for the az agent
`data/endgame_labels.pkl`: 5,718 records in the sp_v4f.pkl format `(planes0-12 uint8 (13,8,8), planes13-15 float32 (3,), z, indices int16, counts int16, rootQ)`, side-to-move perspective. counts = round(200 * softmax(teacher scores / 0.4)); rootQ = tanh(best/3); z = sign(best) if decided (|best| >= 50) else rootQ.
Sources: eg1 + egval (4,912 positions: Kernel + 0-2 random pieces per side, rounds 6-25, teacher depth = rest of turn + 2 full turns, 75% of trivially decided positions dropped) and eg2 (806 positions: Kernel + 1 piece each side, Trace on one side 60% of the time, rounds 6-16, depth = rest of turn + 3 full turns). 48% Magenta to move; 50% are decided wins or losses. Rebuild with `make_endgame_pkl.py`.
Not included: the replayed live-loss positions (`livepos.py`). Its O(n^2) replay was too slow on the loaded machine and was stopped.
`data/endgame_live.pkl`: 396 records in the same format, made by `livepos2.py`. They are the live-loss positions of seeds 12278 a43, 13136 a22, 13365 a25, 11686 a36, 11791 a27 and 12113 a26, plus the 10 positions before each (66 base positions), and 5 random perturbations of each (one piece moved up to 2 squares, or a non-kernel piece removed). Each is labelled by a teacher search to the end of the turn plus 2 full turns, or 3 when each side has at most 1 non-kernel piece.
