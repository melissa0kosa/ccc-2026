# Level 4 deploys

| UTC | model | local evidence |
|---|---|---|
| ~13:51 | stopgap_v1.onnx | 35/40 (88%) vs seat_l3_material, 40/40 vs greedy, 1.1 s cpu/game |
| ~14:01 | hand_v2.onnx | 68/80 (85%) vs stopgap_v1 (agent), 42/48 (88%) vs stopgap_v1 (main arena), cyan 82-83%; validator flaky under load (2/3 accept, dry run 111-186 ms) |
| 14:43 | az_res_v6.onnx | gatekeeper: 138/200 (69%) [62,75] vs hand_v2, cyan 70 / magenta 68, 56/60 vs stopgap, 20/20 greedy, 503 nodes/444 inits, 1.98 s cpu/game |
| 14:53 | az_res_v20.onnx | gatekeeper: 170/220 (77%) [71,82] vs az_res_v6, cyan 77 / magenta 77, 60/60 stopgap, 20/20 greedy, 204 nodes/258 inits, 2.28 s cpu/game |

Live before hand_v2: stopgap_v1 took rating 1120 -> 1481 (rank ~80) in ~13 min.

## Upload gate (adopted 14:10)
- >= 60 alternating-colour games vs the CURRENT seat; Wilson 95% lower bound > 0.55 (>= 41/60, 53/80, 65/100);
  after 16:00 require > 0.60 (>= 44/60). Neither colour below 50%. >= 95% vs greedy.
- No drop > 10 pp vs any gauntlet model (seat_l3_material, stopgap_v1, hand_v1/v2).
- User 14:40: release IMMEDIATELY when a candidate passes all gates and is clearly better (no windows). Cutoffs (revised 14:35, game rate ~0.76/min): +10 pp none after 15:50,
  +20 pp none after 16:20, hard stop 16:40 except rollback. Never end with < 8 games on the live model.
- Each upload costs ~10 games of +-45 rating swing: only upload edges clearly > +50 Elo.
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
| az_res_v20.onnx | 61531c6c73eb | LIVE since 14:53; pinned as champion.onnx |

Live 15:09: az_res_v20 1996 (rank 16), 20 games 14 W (cyan 9/10, magenta 5/10).
