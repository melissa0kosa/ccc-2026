# Small-net deep-learning tricks for the latency-capped ResFast residual (CCC 2026 L4, COLONY: ONE)

Scope: soups/SWA/EMA vs ens.py, in-graph D4 test-time augmentation, distillation, auxiliary heads, loss/data weighting, value-head capacity/calibration, plateau diagnosis. Read: agents/az/reslib.py, azlib.py (train/D4), retrain_fast.py, ens.py, calib.py, RESULTS.md, gate/RECOMMEND.md, reviews/az_res_v6.md, agents/az/log.txt.

**Corrections to the brief (from the repo):**
- **The live seat is no longer v20.** `gate/RECOMMEND.md` (15:18 UTC) released **az_res4f_v2**: 144/220 = 65.5% [59.0, 71.4] vs az_res_v20. The HEAD commit is "deploy az_res4f_v2". Every quick item below should therefore be built on, and gated against, **az_res4f_v2**.
- **The trunk has 2 residual blocks, not 3.** `retrain_fast.py` and `ens.py` build `reslib.ResFast(32, 2)`. 3 blocks is the older `ResModel` on the hand_v2 base.
- **The training recipe:**
  - AdamW at a constant lr of 3e-4, with no schedule, no EMA and no averaging (`azlib.train`).
  - Each step uses one random D4 symmetry per sample.
  - The loss is `lpol + lval` (MSE), weighted 1:1.
  - `torch.manual_seed(1)` is set before the net is built, so **v20, v30, v31 and v32 share the exact same initialisation**. The az_res4f runs share another one.

Local measurements below come from `C:\Users\melis\AppData\Local\Temp\claude\C--Users-melis-Downloads-cloudflight-coding-contest\2537b5b1-8fcc-4ac8-8fd3-b627f7b7b210\scratchpad\tta_soup.py`. It ran on 1 thread on the loaded box (about 39 python processes) at about 15:30 UTC.

## Ranked, test-ready list (before 16:20 vs post-contest)

### Takeaway
- **Before 16:20, in this order:**
  1. Push the one lever that has broken the plateau, **endgame data with z-only, up-weighted value loss**.
  2. **Always add an SWA/EMA tail** to the retrain. It is free at inference and safe in this basin.
  3. Optionally gate a **2-symmetry in-graph TTA** of az_res4f_v2 (+0.6 to 0.8 ms per b8 call, close to the 3 ms cap).
- Soups of the existing v20/v30/v31/v32, and more output ensembles of near-identical members, are **not** worth an arena slot.

### Cited Findings
**Before 16:20 (each about 30-40 min including a 220-game gate; a retrain is 45-120 s on the GPU per `agents/az/log.txt`):**

| # | item | expected gain | latency / graph cost | evidence |
|---|---|---|---|---|
| 1 | **az_res4f_v3 = az_res4f_v2 recipe + more endgame data + per-sample value weight.** <br>• More endgame-start games, including the live-loss seeds 10773 a31, 12113 a29, 11686 a31, 11791 a41 and 11362 a21 (both colours). <br>• `end_frac` raised from 0.26 to about 0.35. <br>• Value loss ×2 on `lowmat` samples (each side ≤1 non-Kernel piece). <br>• z-only targets for endgame samples. | Highest. This is the only lever that has moved the head-to-head: 4f_v2 = 76% vs v20, after 4f_v1 without endgame data scored 50%. | 0 ms, 0 nodes, 0 initializers | RESULTS.md (4f_v1 vs 4f_v2 rows); reviews/az_res_v6.md W1 fix 1; calib.py MSE 0.392/0.413 -> 0.340/0.320 |
| 2 | **SWA/EMA tail.** <br>• Continue from `ckpt_res/az_res4f_v2.pt` (or run inside #1) for 1000-1500 extra steps at constant lr 3e-4. <br>• Average the trainable weights every 25-50 steps (`torch.optim.swa_utils.AveragedModel`, or EMA with decay 0.999). <br>• Then **recompute the BN statistics** on about 20k training positions. | Small but safe: held-out losses improve by about 0.003-0.007 nats policy and 0.001-0.006 MSE value (see the weight-averaging section). | 0 ms, identical graph (278 nodes / 321 initializers) | Izmailov et al. SWA; local v20<->v30 interpolation |
| 3 | **In-graph D4 TTA, S=2** (identity + rot180, or + LR flip). <br>• Run the base once and the residual trunk on 2 symmetric copies batched as 2B. <br>• Undo the policy with one Gather (index `D4_IDX[g] + i*4096`) and average dv. | Unknown sign, but large: on 3,840 endgame positions it changes the prior argmax in 24% of them and v by 0.022 on average. **Must be arena-gated.** | +16 nodes, +10 initializers, +32 KiB per extra symmetry. b8 latency 1.59 -> 2.40 ms (min; median 1.95 -> 3.0). S4 = 3.46 ms and S8 = 7.71 ms, both over budget. | AlphaGo Zero (random symmetry per eval), KataGo (root symmetry averaging); local benchmark |
| 4 | **Value-scale knob.** <br>• `value = tanh(a*(v_h+dv))` with a in {0.8, 1.25}. <br>• Scaling the value scales Q against the fixed c_puct 1.5, so `a` acts as a c_puct proxy. | Speculative; tune by arena only | +1 Mul, +1 initializer, about 0 ms | inference; see the value section |

**Post-contest:**
- Distil the TTA-S8 or ensemble teacher into one net with soft targets.
- Auxiliary training-only heads (KataGo-style ownership/score analogues).
- A symmetry-consistency loss, so the residual becomes equivariant and TTA is no longer needed.
- Hard-example mining from 100- vs 800-sim disagreements.
- A value MLP width ablation (64 -> 128).
- A greedy soup over many same-trajectory fine-tunes.

### Inferences
- **Do 1+2 together as one candidate.** Then decide whether to wrap it with S=2 TTA as a second arm, only if the validator passes it under quiet load and latency allows. Do not stack untested changes into one upload.
- Gate every item against **az_res4f_v2**, not v20.

### Gaps
- No arena numbers exist yet for SWA or TTA on this game. The gains are extrapolated from held-out loss and the literature.

## Weight averaging (soups, SWA, EMA) vs output ensembling (ens.py)

### Takeaway
- **The requirement is a shared basin.**
  - **Same-trajectory checkpoints** (v20 -> v30, same run continued; v20 -> v21 fine-tune) are linearly connected, and their midpoint beats both ends. So **SWA/EMA on the deployed recipe is safe**.
  - **Same-init siblings with a different minibatch order** (v20 vs v32) show a **loss barrier** that BN recompute only partly removes.
- Averaging the existing v20/v30/v31/v32 gives at best about -0.005 nats and stays in the weaker v2-base family. **Skip it.**

### Cited Findings
**Literature:**
- **Model soups need a shared fine-tuning start.**
  - Models fine-tuned from the same pre-trained start "lie in the same basin". Interpolating models trained from scratch can fail completely. There can be "an error barrier between fine-tuned models", mostly at high learning rates, which is why the greedy soup excludes them. — [Wortsman et al. 2022, Model soups](https://arxiv.org/html/2203.05482v3)
  - **Greedy soup:** sort the ingredients by held-out accuracy and add each one only if held-out accuracy does not drop. Because of that ordering it "can be no worse than the best individual model on the held-out validation set". — [Model soups](https://arxiv.org/html/2203.05482v3)
  - **Numbers (CLIP ViT-B/32 ImageNet top-1):** best single model 80.38, uniform soup 79.97 (worse), greedy soup 81.03, output ensemble 81.19. Soups cost no extra inference; ensembles need a pass per member. — [Model soups](https://arxiv.org/html/2203.05482v3); "without incurring any additional inference or memory costs" — [abstract](https://arxiv.org/abs/2203.05482)
- **SWA:**
  - SWA averages points along the SGD trajectory "with a cyclical or constant learning rate", capturing once per epoch for a constant LR. — [Izmailov et al. 2018, SWA](https://ar5iv.labs.arxiv.org/html/1803.05407)
  - **BN:** "we run one additional pass over the data" to recompute BatchNorm statistics for the averaged weights. — [SWA](https://ar5iv.labs.arxiv.org/html/1803.05407)
  - **From pretrained models, 10 extra epochs:** ResNet-50 76.15 -> 76.97, ResNet-152 78.31 -> 78.94, DenseNet-161 77.65 -> 78.44. "SWA is able to achieve comparable or better performance than FGE ensembles with just one model." — [SWA](https://ar5iv.labs.arxiv.org/html/1803.05407)

**Local measurements on our checkpoints:**
- Held-out sets: `games_res_v20x.pkl`, 900 v20 self-play games from 15:12 UTC, unseen by v20/v30/v31/v32 (6,000 sampled positions, z-only value target); and `games_end_holdout1.pkl` (3,840 positions).
- BN was recomputed on 4,096 positions from `games_res_v8`.
- Trunk weight distance to v20: v21 1.79, v30 3.97, v31 3.83, v32 7.76.

| weights | held-out lpol (BN-recomputed) | held-out lval | endgame lval |
|---|---|---|---|
| v20 | 2.0821 | 0.6447 | 0.4070 |
| v30 (same run, 6000 steps) | 2.0784 | 0.6402 | 0.4211 |
| v31 (same data/order, qmix 0.5) | 2.0824 | 0.6414 | 0.4067 |
| v32 (more data, so a different batch order) | 2.0876 | 0.6382 | 0.4057 |
| v21 (fine-tuned from v20 on v21 self-play) | 2.0676 | 0.6326 | 0.4044 |
| **½(v20+v30)** | **2.0756** (better than both) | **0.6387** (better than both) | 0.4112 |
| ½(v20+v31) | 2.0796 (better than both) | 0.6446 | 0.4065 |
| ½(v20+v32) | **2.0925 (worse than both; naive, without BN recompute: 2.1122)** | 0.6456 | 0.3972 |
| ½(v20+v21) | 2.0710 (between) | 0.6364 | 0.4041 |
| uniform soup v20/v30/v31/v32 | 2.0816 (naive 2.1010) | 0.6431 | 0.4015 |

**Output ensemble already tested:** az_res_ens2 (v20+v30) scored 57/120 = 47.5% vs v20, at 239 nodes / 290 initializers and 2.05 ms. — gate/RECOMMEND.md, agents/az/log.txt

### Inferences
- **Lineage:**
  - v20, v30 and v31 descend from the same init (seed 1) and, because N is identical and the same seed drives `torch.randint`, the same minibatch indices. v30 is literally v20's run continued to 6000 steps.
  - v32 shares the init, but its larger N changes every minibatch from step 0.
  - v21 is a fine-tune of v20.
  - This matches the soups/linear-mode-connectivity picture: a shared trajectory gives a convex path; a shared init but different SGD noise gives a barrier.
- **Why ens2 failed:** v20 and v30 are almost the same function. Their held-out losses differ by less than 0.005, so an output ensemble of them adds latency and no diversity. A soup gets the same averaging benefit for free.
- **SWA sketch for retrain_fast.py** (after `train(...)`; about 1 min on the GPU; 0 inference cost):
  ```python
  from torch.optim.swa_utils import AveragedModel
  swa = AveragedModel(net)            # or: multi_avg_fn=get_ema_multi_avg_fn(0.999)
  for i in range(30):                 # 30 x 50 = 1500 tail steps at constant lr
      train(net, opt, buf, 50, qmix=qmix); swa.update_parameters(net)
  m = swa.module
  for mod in m.modules():             # BN recompute: trunk BNs + fbn (base uses buffers, no BN)
      if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
          mod.reset_running_stats(); mod.momentum = None
  m.train()
  with torch.no_grad():
      for i in range(0, 20480, 512): m(obs_batch[i:i+512])   # un-augmented training positions /8.0
  m.eval(); export(m, out)            # same graph, same latency
  ```
  `buf.tensors()` rebuilds the GPU tensors on every `train()` call. For speed, either call it once and inline the loop, or use 10 x 150-step chunks.
- **Greedy-soup selection metric (cheap, about 1 min on CPU):**
  - Held-out lpol and lval on the newest self-play of the *current best* (v20x games now; 4f_v2 games next), plus calib.py endgame MSE.
  - Add an ingredient only if neither held-out loss gets worse.
  - Use it to *rank* soups, not to replace the arena gate.

### Gaps
- Not measured: whether held-out loss differences of about 0.005 nats translate into win rate. The ½(v20+v30) soup was not arena-tested; a ≥55% result is unlikely given its size.
- Not measured: interpolation in the az_res4f family. 4f_v1 and 4f_v2 used different data, so their batch orders differ from step 0 and a barrier is likely, as with v32.

## Test-time D4 symmetry averaging inside the ONNX graph

### Takeaway
- **It is feasible with allowed ops** (Gather, Reshape, Transpose, Concat, ReduceMean) for only +16 nodes and +10 initializers.
- **Only S=2 fits the latency cap** (+0.6 to 0.8 ms per b8 call).
- **The residual is far from equivariant**, so TTA changes the prior substantially. It is a real ensemble with an unknown sign and must be arena-gated.
- **The base is computed once:**
  - v_h and the S/G features are exactly invariant.
  - The v4f base policy has small asymmetric tie-break terms (up to 0.09 logits), which should be kept untouched.

### Cited Findings
**Literature:**
- **AlphaGo / AlphaGo Zero** exploited the rotation and reflection invariance of Go in two ways: training data were augmented with all 8 symmetries, and during MCTS each position was transformed by a randomly selected rotation or reflection before network evaluation, so "the Monte-Carlo evaluation is averaged over different biases". AlphaZero dropped this for chess and shogi because those boards are not symmetric. — [AlphaZero paper (arXiv 1712.01815)](https://arxiv.org/pdf/1712.01815); [AlphaZero preprint](https://discovery-pp.ucl.ac.uk/id/eprint/10069050/1/alphazero_preprint.pdf)
- **KataGo** passes "different rotated/reflected copies of the game-board and average[s] their results in order to obtain a more stable and symmetry-equivariant policy", using a randomly sampled subset of D4 at the root. — [Can Go AIs be adversarially robust? (arXiv 2406.12843)](https://arxiv.org/pdf/2406.12843)
- **The policy must be un-transformed with the correct move-index map;** naive reflection of the policy tensor gives wrong moves. — [Finite Group Equivariant NNs for Games (arXiv 2009.05027)](https://arxiv.org/pdf/2009.05027)
- **Allowed ops** include Gather, Slice, Transpose, Concat, Split, ReduceMean and Mean. Index and shape inputs must be constant initializers, and `Constant` nodes are banned. — research_notes/AI vs AI game model optimization/architecture_inference.md (from submission_policy.py L109-177)

**Local measurement:**

| model | b8 latency, 1 thread (min / median of 5 interleaved rounds, loaded box) | graph | change vs S1 on 3,840 endgame positions |
|---|---|---|---|
| az_res4f_v2 original | 1.59 / 1.95 ms | 278 / 321 | - |
| TTA wrapper S1 | 1.80 / 2.13 | 294 nodes / 331 initializers, parity vs original 0.0 | - |
| **S2 rot180** | **2.41 / 2.97** | 294 / 331, 1066 KiB | mean abs dv 0.022, prior-argmax agreement 0.76 |
| S2 LR flip | 2.40 / 3.02 | same | - |
| S4 | 3.46 / 4.72 | 1131 KiB | 0.027, 0.71 |
| S8 | 7.71 / 8.78 | 1261 KiB | 0.026, 0.70 |
| az_res_v20 original / S2 / S4 / S8 | 1.01 / 1.96 / 3.37 / 6.34 (min) | 220 / 268 | S2 agreement 0.51, S8 0.40 |

**Base equivariance under the training D4 map:**
- v2 base: exact for the policy, value and S/G.
- v4f base: value and S/G exact; policy max |diff| 0.05-0.09 (the tie-break buffers `tieb`/`tieb2` in fast_base).

### Inferences
- **Implementation** (validated: S1 reproduces the original exactly, and the same index maps make the v2 base exactly equivariant). Export with `azlib.export`:
  ```python
  class TTA(nn.Module):              # r = loaded ResFast; syms e.g. [0, 3] (identity + rot180)
      def __init__(self, r, syms):
          super().__init__(); self.r, self.S = r, len(syms)
          self.register_buffer("xin", torch.from_numpy(np.concatenate([D4_INV[s] for s in syms])))
          self.register_buffer("pidx", torch.from_numpy(np.concatenate([D4_IDX[s] + i*4096 for i, s in enumerate(syms)])))
      def forward(self, x):
          r, S = self.r, self.S
          ph, vh, Sf, G = r.base(x); vh = vh.reshape(-1)                         # base ONCE
          xs = x.reshape(-1, 16, 64)[:, :, self.xin].reshape(-1, 16, S, 64).transpose(1, 2).reshape(-1, 16, 8, 8)
          h = r.blocks(r.stem(xs))                                               # trunk on B*S
          f = r.pf(h).reshape(-1, r.k, 64).transpose(1, 2); t = r.pt(h).reshape(-1, r.k, 64)
          dp = (torch.matmul(f, t) * r.scale + r.p64(h).reshape(-1, 64, 64)).reshape(-1, S * 4096)
          dp = dp[:, self.pidx].reshape(-1, S, 4096).mean(1)                     # one Gather un-permutes all
          sg = r.fbn(torch.cat([Sf, G], 1)); sg = torch.cat([sg[:, None, :]] * S, 1).reshape(-1, sg.shape[1])
          z = torch.cat([r.v1(h).flatten(1), h.mean(dim=(2, 3)), sg], 1)
          dv = r.vfc2(F.relu(r.vfc1(z))).reshape(-1, S).mean(1)
          return ph + dp, torch.tanh(vh + dv)
  ```
  - The S/G features are invariant, so they are reused for every copy (no base recompute).
  - **Cost scales with the trunk only:** about +0.4 ms per extra symmetry at b8 on 4f. Batching the copies (b16 instead of 2 x b8) gets better than linear scaling.
- **Why it can matter here:**
  - The residual is trained with one random symmetry per sample for about 5 epochs (3000 x 512 / 299k), so it is not equivariant: v20 changes argmax in half of the endgame positions under one symmetry.
  - Averaging is then a free 2-member ensemble with diverse "views", unlike ens2.
  - It also smooths the prior, which may hurt where the sharp prior helped (review W5: prior top-1 median 0.24). That is why the arena decides.
- **S=2 latency margin:** 2.4 ms min and 3.0 ms median under load. Check the official validator on a quiet box before any upload, because the "too slow to check" rejections were load artefacts.

### Gaps
- I found no published Elo number for 8-fold symmetry averaging versus a single symmetry in AlphaGo, AlphaGo Zero or KataGo. The sources describe the mechanism, not a measured gain.
- KataGo's exact config keys (e.g. `rootNumSymmetriesToSample`) could not be verified this session.

## Distillation of an ensemble or TTA back into one net

### Takeaway
- It is cheap only if the teacher's soft targets are computed on positions already on disk. It does not fix a data problem.
- The team's distill students all plateaued at 42-54% vs v20.
- **Post-contest**, unless TTA-S2 wins its gate and fails the latency check.

### Cited Findings
- Distillation can "compress the knowledge in an ensemble into a single model which is much easier to deploy", improving a production acoustic model. — [Hinton, Vinyals, Dean 2015](https://arxiv.org/abs/1503.02531)
- KataGo's auxiliary soft policy target (T = 4, weighted 8x the normal policy target) "appears to greatly improve the speed of learning of the policy". — [KataGo methods doc](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
- Team students: distill_v5s8 50.0%, distill_v5hf 47.5%, distill_v4s8 42.2% vs v20. — gate/RECOMMEND.md

### Inferences
- **Sketch:**
  1. Teacher = TTA-S8 of az_res4f_v2 (equivariant-ish, about 4x slower, fine offline on the GPU).
  2. Relabel all 299k training positions with teacher logits (dp, restricted to the legal `idx`) and the teacher's dv. On the GPU this takes seconds.
  3. Student = the same ResFast(32, 2), initialised from 4f_v2 (same basin).
  4. Loss = 0.5·CE(MCTS visits) + 0.5·KL(teacher softmax at T = 1-2 over legal moves) + value MSE to 0.5·z + 0.5·v_teacher.
  5. Train 1500 steps.
- The result keeps the original latency and approximates the TTA gain.
- This adds no new game information, so it is capped by the teacher.

### Gaps
- No measurement exists of how much of a symmetry-ensemble gain survives distillation into a 2-block net.

## Auxiliary heads and targets

### Takeaway
- KataGo's evidence is strong (ownership and score targets were worth 1.65x training efficiency), but it comes from long runs.
- Training-only heads cost **0 ms and 0 exported initializers**, because they are dropped from the export wrapper.
- Coding plus relabelling plus a gate takes more than 40 min, so it is **post-contest**.

### Cited Findings
- Removing the ownership and score targets dropped KataGo from 1329 to 1139 Elo at equal compute, about 1.65x the training time to reach the same Elo. Ownership loss weight 1.5/b²; score pdf/cdf 0.02 each; opponent policy 0.15 ("for regularization only"). — [KataGo paper, Wu 2019](https://ar5iv.labs.arxiv.org/html/1902.10565)
- Short-term value/score targets: nets "seem to train slightly faster and achieve slightly better value and score loss on the main head" (only partially re-tested). — [KataGo methods doc](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)

### Inferences
**Targets that can be derived from saved games without new self-play** (each game dict has `obs` per position and `z`; the last obs is the final position):
- (a) **End-of-game occupancy map** (the ownership analogue): per square, own Kernel / own piece / enemy Kernel / enemy piece at game end. Head `nn.Conv2d(32, 4, 1)` on `h`, BCE, weight about 0.1-0.2. The spatial target must be permuted with the same `inv[sy]` Gather as `x` in `azlib.train`.
- (b) **Final material difference** and **remaining game length** `(n - i)/100`: a Linear on the value vector `z`, MSE weight about 0.1.
- (c) **Bare-Kernel race winner / final Kernel ring depth difference** on `lowmat` positions only: directly targets W1.
- Export with the same `forward` (heads unused), so ONNX drops them.

Expected effect on a 3000-step budget: modest. The value of (a) and (c) is that they force the trunk to represent ring geometry and guard reach, which the review says the value misses.

### Gaps
- No evidence was found on auxiliary-target gains at very short budgets (a few thousand steps) or for 2-block nets.

## Loss and data weighting

### Takeaway
- **Data reweighting toward low-material endgames is the proven lever.**
  - Next increments: a higher endgame fraction, live-loss seeds, and a per-sample value weight on `lowmat`.
  - Keep z-only targets there: Q is biased exactly in these positions.
- Focal or hard-example mining needs 800-sim labels, so it is post-contest.
- Policy label smoothing is pointless: the targets are already soft visit distributions.

### Cited Findings
- 4f_v2 (endgame starts = 26% of samples, outcome-only value targets) vs 4f_v1 (none): 76% vs 50% against v20. Endgame MSE: MAG 0.392 -> 0.340, CYAN 0.413 -> 0.320. — agents/az/RESULTS.md
- **Live blind spot:**
  - Mean raw value in positions the side to move later lost: MAGENTA -0.26/-0.27/-0.20 (v6/v20/v4f) vs CYAN -0.86/-0.91/-0.83.
  - In the endgame losses, 100- and 800-sim search agree on the losing moves, so 800-sim labels share the blind spot. — reviews/az_res_v6.md §3, §5
- **Focal loss** down-weights well-classified examples, (1-p_t)^γ·CE. — [Lin et al. 2017](https://arxiv.org/abs/1708.02002) (not re-fetched this session)

### Inferences
**Sketch, a 5-line change to `azlib.train`:**
- Pass a per-sample weight `w = 1 + 1.0*lowmat` (lowmat from the planes: count of non-Kernel own/enemy pieces ≤ 1) and use `lval = (w * (v - z[b])**2).mean()`.
- Optionally use value loss weight 1.5 overall: the value head trains on 1 scalar per sample against about 30 policy targets.
- Use `end_frac` 0.35 plus the five live-loss seed positions (both colours) via gen_end.py.
- Train lr 3e-4 for 3000 steps plus the SWA tail.
- Check with calib.py before the arena: MAG lost mean v should move clearly below -0.5.

**Hard-example mining (post):**
- Relabel decisions where the 100- and 800-sim choices differ (reviews v6_scan: 62/219 = 28%) with 800-sim visits, and upweight them ×3.
- Not useful for W1, because 800-sim shares the bias there. It needs z-only endgame games instead.

### Gaps
- Not measured: the optimal endgame fraction, or whether `lowmat` up-weighting degrades middlegame play. Watch the stopgap/greedy guards.

## Value-head capacity and calibration

### Takeaway
- The 64-unit value MLP is **probably not** the bottleneck. The same head improved sharply once endgame data arrived, and the error is a *colour-conditional bias*.
- A single temperature cannot fix a colour-conditional bias.
- A scalar value scale is still a cheap c_puct-like knob, worth at most one arena arm.

### Cited Findings
- Same architecture and head: endgame MSE fell 0.392 -> 0.340 (MAG) and 0.413 -> 0.320 (CYAN) when endgame data was added (4f_v1 -> 4f_v2). — agents/az/RESULTS.md
- **Train vs held-out:**
  - Train lval is 0.36-0.42 against 0.7z + 0.3Q targets (log.txt).
  - Held-out lval on fresh v20 games is 0.633-0.646 against z for every v2-family checkpoint (local table above).
  - This is a distribution-shift or generalisation gap, not underfitting.
- calib.py only reports MAG/CYAN lost/won means and MSE. It does not fit a temperature. — agents/az/calib.py

### Inferences
**Quick checks (each at most 5 min, no arena):**
- (i) **Retrain 4f_v2 with `vfc1` 64 -> 128.** This changes only tensor sizes: 0 extra initializers and about +0.01 ms. If calib.py endgame MSE drops more than 0.02, capacity matters.
- (ii) **Fit `a` in `tanh(a·pre)`** on games_end_holdout by 1-D search on MSE.
  - If a ≈ 1, skip it.
  - Otherwise, a > 1 sharpens Q differences, which acts like a lower effective c_puct in the referee PUCT (c_puct 1.5, FPU 0).
  - Cost: 1 Mul plus 1 initializer.
- **A conditional fix** (`b·lowmat·is_magenta` added to the pre-activation) is better done in the base features, as the review recommends (W1 fix 2).

### Gaps
- Not measured: the effective-c_puct effect of value scaling in this referee. It needs an arena run.

## Why retrains plateaued at about 50% vs v20, and the quickest diagnostic

### Takeaway
- **The plateau was data, not evaluation saturation or capacity.**
  - v30, v31 and v32 are functionally the same model as v20: held-out losses agree within about 0.005-0.013.
  - The arena *can* resolve real gains: 4f_v2 scored 65.5-76% vs v20 once new information was added (endgame starts, sp_v4f, v20 games, a better base).
- **Quickest diagnostic: about 1 min of held-out loss on the newest self-play of the current best, plus calib.py, before spending 220 arena games.**

### Cited Findings
- **Data:** v20, v30 and v31 were trained on the identical 133,651 positions and v32 on 154,145; 4f_v2 on 299,345, including 26% endgame starts. — agents/az/log.txt
- **Held-out on 900 fresh v20 games** (v20 / v30 / v31 / v32):
  - lpol 2.082 / 2.078 / 2.082 / 2.088
  - lval 0.645 / 0.640 / 0.641 / 0.638
  - Arena vs v20: 53% / 46% / 48%. — local tta_soup.py run; RESULTS.md
- The same arena method measured 4f_v2 at 144/220 = 65.5% [59.0, 71.4] and 152/200 = 76% vs v20, so it is not saturated. — gate/RECOMMEND.md; RESULTS.md
- The positions come from v4-v8 self-play (older, weaker searches). More steps (v30) or a different value mix (v31) re-fit the same targets: train lpol 2.183 / 2.166 / 2.182. — agents/az/log.txt

### Inferences
**Pre-arena screen for any candidate:**
1. Held-out lpol/lval on the newest best-model self-play.
2. calib.py endgame MAG/CYAN.
3. The 268 live low-material positions (reviews/v6_lowmat.py).

If none improves on the parent by about 0.01 or more, do not spend the 220-game gate.

### Gaps
- Not validated: the correlation between held-out loss deltas and arena win rate. There are only about 6 data points, and 4f_v2 changed the base too, so its loss is not directly comparable.
