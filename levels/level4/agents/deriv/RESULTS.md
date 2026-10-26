# Derivations of az_res4f_v2 (inference-time / graph-only), screened vs az_res4f_v2

Opponent: az_res4f_v2.onnx (md5 f0329dfae66e629c61accd855815f2a2). 100 sims, alternating colours, fresh seeds.
Every run (with sha256 of both files) is in arena_log.jsonl. CPU = per game, both sides (champion vs itself is about 1.3 s; the box was loaded).
Builders: derive.py (TTA batch stacking, policy/value temperature, warm, residual scale), rtta.py (residual-only symmetry averaging, torch export).

| variant | what | games | wins | rate | Wilson 95% | C / M | CPU/game | verdict |
|---|---|---|---|---|---|---|---|---|
| deriv_4f2_tta2 | full TTA {id, LR flip} | 100 | 52 | 52% | [42,62] | 54/50 | 2.3 s | neutral (handed to Checker B) |
| deriv_4f2_tta4 | full TTA {id, LR, UD, rot180} | 100 | 55 | 55% | [45,64] | 52/58 | 4.2 s | too costly (~3.4 s our side), handed to Checker B |
| deriv_4f2_w6 | round-1 logits x6 | 300 (100+200) | 156 | 52.0% | [46.4,57.6] | 100: 68/52, 200: 53/43 | 1.8-2.0 s | neutral |
| deriv_4f2_w4 | round-1 logits x4 | 100 | 42 | 42% | [33,52] | 42/42 | 1.8 s | no |
| deriv_4f2_pt08 | policy logits x0.8 | 100 | 54 | 54% | [44,63] | 48/60 | 1.7 s | neutral |
| deriv_4f2_pt125 | policy logits x1.25 | 100 | 42 | 42% | [33,52] | 40/44 | 1.8 s | no |
| deriv_4f2_pt15 | policy logits x1.5 | 100 | 50 | 50% | [40,60] | 50/50 | 1.9 s | no |
| deriv_4f2_vt08 | value tanh-input x0.8 | 300 (100+200) | 138 | 46.0% | [40.4,51.7] | 100: 52/66, 200: 36/43 | 1.4-1.9 s | no (59/100 was noise) |
| deriv_4f2_vt07 | value tanh-input x0.7 | 100 | 49 | 49% | [39,59] | 48/50 | 1.8 s | no |
| deriv_4f2_vt125 | value tanh-input x1.25 | 100 | 52 | 52% | [42,62] | 58/46 | 1.7 s | neutral |
| deriv_4f2_vt08w6 | vt08 + round-1 x6 | 100 | 49 | 49% | [39,59] | 56/42 | 1.6 s | no |
| deriv_4f2_rs125 | residual (dp, dv) x1.25 | 100 | 46 | 46% | [37,56] | 46/46 | 1.7 s | no |
| deriv_4f2_rs075 | residual (dp, dv) x0.75 | 100 | 46 | 46% | [37,56] | 44/48 | 2.0 s | no |
| deriv_4f2_rtta2 | residual-only TTA {id, LR}; base run once | 120 | 63 | 52.5% | [44,61] | 55/50 | 2.0 s | neutral, below the 58% extension bar |
| deriv_4f2_rtta4 | residual-only TTA {id, LR, UD, rot180} | - | - | - | - | - | 3.44 ms/b8 call (champion 1.56) | built and parity-checked, not screened (stopped) |
| deriv_4f2_vt09, rs15, rsv125, rsp125, tta8 | built, not screened by me | - | - | - | - | - | - | gatekeeper: rsv125 141/240 = 58.8%, rsp125 48/100 |

Exactness checks:
- TTA graphs equal numpy averaging of the champion (max |dpol| 0, |dval| 1e-7) and are equivariant under their group (<= 2e-6).
- rtta2/rtta4 equal residual averaging (|dpol| <= 2e-6, |dval| 3e-7). The 4f base value and S/G features are invariant under the
  symmetries (|dS| 2e-7), but the base policy is not exactly equivariant (|dph| up to 0.13). rtta2 changes the prior argmax in 196/1693
  positions, rtta4 in 264/1693.
- The temperature, warm and residual-scale variants change only the intended tensor (policy unchanged to 0.0 where expected).

Conclusion: no finalist. Every graph-only tweak of az_res4f_v2 is within noise of 50%, and no pooled Wilson lower bound exceeds 0.55.
Not done: validator and greedy runs (no candidate qualified).
Process note: the stopped first chain (TaskStop at ~15:35:45) still launched its last command (w6 +200 games), which overlapped the
rtta2 screen for about 2 minutes, so 8 workers ran during that window. Both results are valid, only slower.
