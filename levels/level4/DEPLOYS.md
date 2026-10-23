# Level 4 deploys

| UTC | model | local evidence |
|---|---|---|
| ~13:51 | stopgap_v1.onnx | 35/40 (88%) vs seat_l3_material, 40/40 vs greedy, 1.1 s cpu/game |
| ~14:01 | hand_v2.onnx | 68/80 (85%) vs stopgap_v1 (agent), 42/48 (88%) vs stopgap_v1 (main arena), cyan 82-83%; validator flaky under load (2/3 accept, dry run 111-186 ms) |

Live before hand_v2: stopgap_v1 took rating 1120 -> 1481 (rank ~80) in ~13 min.

## Upload gate (adopted 14:10)
- >= 60 alternating-colour games vs the CURRENT seat; Wilson 95% lower bound > 0.55 (>= 41/60, 53/80, 65/100);
  after 16:00 require > 0.60 (>= 44/60). Neither colour below 50%. >= 95% vs greedy.
- No drop > 10 pp vs any gauntlet model (seat_l3_material, stopgap_v1, hand_v1/v2).
- Bundle small gains; cutoffs: +5 pp none after 15:45, +10 pp none after ~16:15, +20 pp none after ~16:30;
  hard stop 16:45 except rollback of a broken model. Never end with < 8 games on the live model.
- Validator must ACCEPT (re-run on rejection: under load the 2 s check is flaky).
