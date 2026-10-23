# Level 4 deploys

| UTC | model | local evidence |
|---|---|---|
| 13:54 | stopgap_v1.onnx | 35/40 (88%) vs seat_l3_material, 40/40 vs greedy, 1.1 s cpu/game |
| 14:07 | hand_v2.onnx | 68/80 (85%) vs stopgap_v1 (agent), 42/48 (88%) vs stopgap_v1 (main arena), cyan 82-83%; validator flaky under load (2/3 accept, dry run 111-186 ms) |

Live before hand_v2: stopgap_v1 took rating 1120 -> 1481 (rank ~80) in ~13 min.
