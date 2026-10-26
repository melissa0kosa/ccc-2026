# hand agent results (100 sims, alternating colours, Wilson 95% CI)

Model = fixed feature extractor (exact capture-threat maps with blocking, kernel guards,
firewall next-ring, side-to-move terms) -> linear -> tanh; hand policy prior.
Code: `levels/level4/agents/hand/hand.py`, export: `export_model.py`.

| model | vs seat_l3_material | vs greedy | vs prev best | cpu/game (both) | notes |
|---|---|---|---|---|---|
| hand_v1 | 57/60 = 95% [86,98] cyan 93% / mag 97% | 59/60 = 98% [91,100] cyan 97% / mag 100% | - | 1.0 s (vs seat) | hand-set weights, 1.6 ms/call b8 |
| hand_v2 | 58/60 = 97% [89,99] cyan 93% / mag 100% | 39/40 = 98% [87,100] cyan 100% / mag 95% | vs v1: 50/60 = 83% [72,91] cyan 87% / mag 80%; **vs stopgap_v1: 68/80 = 85% [76,91] cyan 82% / mag 88%** | 0.9-1.4 s | + 2-action kernel-strike features (quiet move -> attack kernel, blocking-aware), firewall escape/u2 terms; value weights Texel-fitted on 600 v1 self-play games (21.7k positions, val acc 0.70); policy params fitted to root visit counts. 3.5 ms/call b8 |

v1 vs stopgap_v1: 31/60 = 52% [39,64] cyan 50% / mag 53%.
End causes (v2 vs stopgap, 80 games): wins 60 capture + 8 bare-kernel; losses 11 capture + 1 firewall(r8).
| hand_v2fast | = hand_v2 (output-identical: max abs diff value 2.7e-7, legal-move logits 1e-6) | | | | lean FastNet graph (`fast.py`): 167 nodes / 225 initializers, dry run 109 ms vs 196 ms |
| **hand_v3** | 60/60 = 100% [94,100] | 40/40 = 100% [91,100] | **vs v2: 83/120 = 69% [60,77] cyan 68% / mag 70%**; vs stopgap_v1: 68/80 = 85% [76,91] cyan 88% / mag 82% | 0.9-1.3 s | Texel refit on v1+v2 self-play (53k pos, val acc 0.718) + Cyan-specific firewall terms; FastNet export 167 nodes / 225 inits, dry run 79 ms |

A/B vs hand_v3 (200 games each, none better):
- v3a soft x0.5 (hard kept), bias 0: 84/200 = 42% [35,49]
- v3b soft x0.5, hard win_now 3 / win2 2 / enK_burn 3: 85/200 = 42% [36,49]
- v3c all x1.3: 98/200 = 49% [42,56]
- v4a policy + 2-action Kernel danger (k_d2 3), Kernel escape when threatened (2), guard bonus (1.5), tie-breaks: 104/200 = 52% [45,59]
- v4b = v4a with prior x1.5: 103/200 = 52% [45,58]

Cross-check v3 vs v2 with c1.arena (same code path as myarena): 74/120 = 62% [53,70] cyan 48% / mag 75%. Pooled with myarena: 157/240 = 65%.

### v4 bundle (fast4.py = FastNet4; export4.py; fit4.py)
Graph 238 nodes / 286 initializers, dry run 100-131 ms. Prior fixes from the hand_v2 review: guard-aware Kernel-attack bonus (ek_guard 1.5),
2-action enemy reach map (both sides, second ray pass) -> Kernel destination danger (k_d2 3) + piece danger2 0.25, Kernel escape when threatened (2),
guard bonus (1.5), early Kernel evacuation k_fw2 0.6*ownK*N1*u2, opening sharpening x(1+0.5*warm), tie-breaks 0.05.
New value features: ownK_g1_t2, enK_g1_t2, own_hang2_<t>, en_hang2a2_<t> (2-action material strikes).
- hand_v4f = v4 bundle + value refit on v1+v2+v3 self-play (84k pos, val acc 0.752): vs v3 109/200 = 55% [48,61] cyan 59% / mag 50%; 1.9 s cpu/game (both)
- hand_v4p = v3 value weights + v4 prior + g1_t2 -0.4/+0.4 (no refit): vs v3 76/200 = 38% [32,45]  (worse)
- hand_v4p is worse than v3 (38%), so the v4 prior needs the refitted value; don't ship prior changes without a refit.
- **hand_v4f vs hand_v2 (sha 8100d416 vs e19e3203): myarena 147/240 = 61% [55,67] cyan 58% / mag 64%; c1.arena 123/200 = 62% [55,68] cyan 57% / mag 66%; pooled 270/440 = 61.4%.**
- Ablations vs v4f (200 games each, all within noise): no opening sharpening 52%, no Kernel-danger/escape/guard 52%, ek_guard -> ek_val 55%, no k_fw2 + danger2 48%.
- hand_v4g = v4f with Kernel danger off in the shrink round (kd_u1off 1): exported, not yet tested.
Source for rebasing: `fast4.py` FastNet4(weights, bias, pol) with `data/w_v4f.json`; the torch module reproduces hand_v4f.onnx (max |dv| 3e-7, legal logits identical).
Arena log with sha256: `data/arena_log.jsonl`.

### v5 attempts (fast5.py = v4 + shot-only threat maps own_shot/en_shot + own_hang2 x a2 gate; fit5.py: L2 toward init, blended targets)
Data: sp_v4f.pkl = 600 hand_v4f self-play games at 200 sims, 30,465 positions, with root search Q (rec[5], side-to-move view).
- hand_v5 (target 0.7*q + 0.3*z, all data 115k pos; first-fit val sign-acc 0.805): vs v4f 93/200 = 46% [40,53] cyan 44% / mag 49%; 8 firewall-r8 losses vs 2
- hand_v5z (outcome-only target, same features): vs v4f 101/200 = 50% [44,57] cyan 47% / mag 54%
=> neither beats v4f; hand_v4f stays the best hand base.

### az_res4f_v2_eg.onnx: exact gated bare-Kernel endgame override on az_res4f_v2 (eg_override.py, eg_verify.py)
md5 e8d97daac3ea688333db01b2529f2602 (base az_res4f_v2 f0329dfae66e629c61accd855815f2a2). 396 nodes / 421 initializers, validator ACCEPTED (157.5 ms dry run).
Latency (1 thread): b1 0.48 vs 0.37 ms, b8 2.02 vs 1.61 ms, b16 3.70 vs 3.54 ms.
- Parity: 6000 real self-play positions (sp_v4f + sp_v3), 263 inside the gate; outside the gate value and policy are bit-identical (0 diffs).
- Brute force vs env (every legal action stepped, 3000 generated positions with al=1, 0-2 locked rings):
  A (bare K vs K+1): 262 win / 949 no-win, 0 value errors, 0 veto errors (veto = exactly the immediately-losing actions);
  B (K vs K): 701 win / 500 lose, 0 value errors, every action terminal;
  gate-off (Magenta to move, urgency 1): 588 positions bit-identical to the base.
- Arena az_res4f_v2_eg vs az_res4f_v2 (my arena, 2 workers): 149/300 = 50% [44,55] cyan 49% / mag 50%, 1.9 s cpu/game (both sides). As expected, the gated positions are rare.
- Reusable wrapper: `eg_wrap.py <src.onnx> <dst.onnx> [n_brute]` (build + validator + parity + brute force, prints PASS/FAIL + md5).
- az_res4f_v3_eg.onnx (src az_res4f_v3 md5 5af5b4c1dea2813a0c4ecd2abe140b06, 15:32:36 UTC) -> md5 c284f9c93fba3f77ec5f7aa7e7df6328: PASS.
  Validator ACCEPTED (136.7 ms), 396 nodes / 421 initializers. Parity on 6000 real positions (263 gated): 0 diffs outside the gate.
  Brute force over 600 positions: A 45 win / 189 no-win, B 142 / 92, gate-off 132, all 0 errors.
- Watcher `eg_watch.sh` (background, 1 h): auto-wraps any new or re-exported az_res4f_v3*.onnx -> *_eg.onnx, log in data/eg_watch.log.
