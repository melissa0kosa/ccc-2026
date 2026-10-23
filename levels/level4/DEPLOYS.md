# Level 4 deploys

| UTC | model | local evidence |
|---|---|---|
| ~13:51 | stopgap_v1.onnx | 35/40 (88%) vs seat_l3_material, 40/40 vs greedy, 1.1 s cpu/game |
| ~14:01 | hand_v2.onnx | 68/80 (85%) vs stopgap_v1 (agent), 42/48 (88%) vs stopgap_v1 (main arena), cyan 82-83%; validator flaky under load (2/3 accept, dry run 111-186 ms) |
| 14:43 | az_res_v6.onnx | gatekeeper: 138/200 (69%) [62,75] vs hand_v2, cyan 70 / magenta 68, 56/60 vs stopgap, 20/20 greedy, 503 nodes/444 inits, 1.98 s cpu/game |
| 14:53 | az_res_v20.onnx | gatekeeper: 170/220 (77%) [71,82] vs az_res_v6, cyan 77 / magenta 77, 60/60 stopgap, 20/20 greedy, 204 nodes/258 inits, 2.28 s cpu/game |
| 15:19 | az_res4f_v2.onnx | gatekeeper: 144/220 (65.5%) [59.0,71.4] vs az_res_v20, cyan 69 / magenta 62, 20/20 greedy, 58/60 stopgap, 278 nodes/321 inits, 1.28 s cpu/game; residual on hand_v4f |
| 15:49 | deriv_4f2_rsv125.onnx | USER DECISION (slight edge): 141/240 (58.8%) [52.4,64.8] vs az_res4f_v2 (below the 0.55 LB bar); validator ok, greedy 20/20, stopgap 55/60, 1.77 s cpu/game. Rollback rule: >75 below v2 after 20 games -> rollback_az_res4f_v2 |
| 15:57 | az_res4f_v6.onnx | GPU sprint (312k pos, oracle/endgame/live data, MAGENTA x2, bf16, SWA): vs rsv125 69/100 (az) + 74/100 (main, fresh seeds) = 143/200 (71.5%) [64.9,77.3], cyan 75 / magenta 68; validator ok; 1.9-2.1 s cpu/game |
| 16:19 | az_res4f_s2.onnx | Sprint-2 (1259 fresh v6 self-play games + fine-tune from v6, ~11 passes): vs v6 65/100 (sprint) + 63/100 (main, fresh) = 128/200 (64%) [57,70], cyan 64 / magenta 64; validator ok; chosen over v9 (221/400 = 55%) |
| 16:23 | az_res4f_x3.onnx | Sprint-3 (5,753 v6@800-sim labels, count^4 targets, x3 on disagreements, value distilled to v6): vs s2 68/100 (sprint) + 70/100 (main) = 138/200 (69%) [62,75]; gauntlet greedy 10/10, stopgap 20/20, hand_v4f 20/20, v20 25/30, v2 35/40, v6 32/40; validator ok; 1.8 s cpu/game |
| 16:29 | az_res4f_x3_eg.onnx | USER DECISION: x3 + exact bare-Kernel override (eg_wrap.py). Bit-identical to x3 outside gate (6000 pos, 0 diffs), brute force 0 errors, validator ok. At 22402 a34: x3 picks suicide E3->F2 5/5 (value +0.61), x3_eg picks E3->D4 5/5 (value -0.95) |

Live before hand_v2: stopgap_v1 took rating 1120 -> 1481 (rank ~80) in ~13 min.

## Upload gate (adopted 14:10)
- >= 60 alternating-colour games vs the CURRENT seat; Wilson 95% lower bound > 0.55 (>= 41/60, 53/80, 65/100);
  after 16:00 require > 0.60 (>= 44/60). Neither colour below 50%. >= 95% vs greedy.
- No drop > 10 pp vs any gauntlet model (seat_l3_material, stopgap_v1, hand_v1/v2).
- User 14:40: release IMMEDIATELY when a candidate passes all gates and is clearly better (no windows). Cutoffs (revised 14:35, game rate ~0.76/min): +10 pp none after 15:50,
  +20 pp none after 16:20, hard stop 16:40 except rollback. Never end with < 8 games on the live model.
- Each upload costs only ~10-12 games (8-15 min) of +-35-85 swing, no systematic loss (f7, 15:12):
  the gate alone (Wilson LB > 0.55) suffices mid-contest; prefer uploading once the live model has
  settled (steps alternating sign). Plan with 0.8 games/min.
  (15:09: the earlier "2 swap forfeits" were a download artefact - older matches lose their protocol in later
  downloads; score field 1 = CYAN won is the reliable result.)
- Live read while climbing (all steps same sign): use performance estimate, not raw rating.
- Validator must ACCEPT (re-run on rejection: under load the 2 s check is flaky).

## Ledger (sha256, first 12)
| model | sha256 | status |
|---|---|---|
| stopgap_v1.onnx | 5be9a9fa78ae | retired 14:01 |
| hand_v2.onnx | e19e320340c3 | retired 14:43 at 1773 (rank 34); kept as rollback_hand_v2.onnx |
| az_res_v6.onnx | deb67cc5f367 | retired 14:53 at 1907 (rank 17); kept as rollback_az_res_v6.onnx |
| az_res_v20.onnx | 61531c6c73eb | retired 15:19 at 1957 (rank 23; settled ~1990-2015); kept as rollback_az_res_v20.onnx |
| az_res4f_v2.onnx | bca01a26679d | retired 15:49 at 2015 (rank 24; peak 2114 rank 14); kept as rollback_az_res4f_v2.onnx |
| deriv_4f2_rsv125.onnx | bfea13e9a52a | retired 15:57 at 2009 (rank 27, ~8 min live); kept as rollback_rsv125.onnx |
| az_res4f_v6.onnx | 2af4b513317b | retired 16:19 at 2083 (rank 21; peak 2234 rank 13); kept as rollback_az_res4f_v6.onnx |
| az_res4f_s2.onnx | 88bdbe75e688 | retired 16:23 at 2008 (rank 29, ~5 min live); kept as rollback_az_res4f_s2.onnx |
| az_res4f_x3.onnx | d6488e7d359f | retired 16:29 at 1896 (rank 39; 0/2 live, lost 22402 to bare-Kernel suicide capture); kept as rollback_az_res4f_x3.onnx |
| az_res4f_x3_eg.onnx | 3a7823c21a1f | LIVE since 16:29 (user decision); pinned as champion.onnx |

Live 15:09: az_res_v20 1996 (rank 16), 20 games 14 W (cyan 9/10, magenta 5/10).
Settled levels (f7): hand_v2 ~1770, az_res_v6 ~1905-1940, az_res_v20 ~1990-2015 (settled ~15:05).
Live 15:32: az_res4f_v2 2114 (rank 14), 16 games 10 W (cyan 6/8, magenta 4/8). Round-4 gate: no v2 derivative passes (best rsv125 58.8% [52,65]).
Quick gauntlet az_res4f_v6 (16:01): greedy 10/10, stopgap 20/20, hand_v4f 18/20, az_res_v20 25/30, az_res4f_v2 28/40 (magenta 55%); validator ok.
Live 16:08: az_res4f_v6 2234 (rank 13, best so far), 8 games 6 W (cyan 3/4, magenta 3/4). rsv125 had 5/8 in its 8 min. v7 (15k steps) vs v6: 54/100 [44,63] -> not shipped.
x4 (16:33; 12,920 new x3@800 labels, 17,948 total): gauntlet greedy 10/10, stopgap 20/20, hand_v4f 18/20, v20 30/30, v2 35/40, v6 32/40, s2 35/40, x3 56/100 [46,65]. USER DECISION 16:35: keep x3_eg live (final); x4 not uploaded.
