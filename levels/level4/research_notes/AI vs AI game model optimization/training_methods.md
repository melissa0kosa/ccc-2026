# Compute-efficient policy/value training for COLONY: ONE (1-3 h, 1 GPU, slow Python engine, shared CPU)

Context used below (from local files, read-only):
- Referee search is PUCT, `Q + c_puct * P * sqrt(sum N + 1) / (1 + N)`, with **unvisited Q = 0** (FPU 0) and virtual loss subtracted from W. Dirichlet alpha 0.3 / eps 0.15, mixed in *after* normalisation, plus temperatures (1.0, 0.5) in 4-action blocks, both only during the 8-action warmup. Source: `levels/level4/data/colony-one-env/colony_one/mcts.py` (lines ~158-190, 333-355).
- Submission limits: 5 MiB ONNX, opset 13-18, match budget 15 s CPU / 20 s total / 512 MiB. Validator checks batch 1 and 16, and search calls up to batch 8. Source: `docs/submission.md`.
- Canonical view: the Magenta board is rotated 180 degrees. Planes 0-5 own, 6-11 enemy, 12 locked, 13 actions_left/2, 14 firewall urgency, 15 Magenta-to-move. Source: `docs/observation.md`.
- Level 4 start (canonical, Cyan to move): own Kernel (7,0), Ping (4,1), Daemon (7,2), Exploit (6,3), Trace (5,0), Ice (6,1). Enemy Kernel (0,7), Ping (3,6), Daemon (0,5), Exploit (1,4), Trace (2,7), Ice (1,6). Source: `levels/level4/level-4_scene.json` (planes 0-11).
- Team az log: the bootstrap net trained to `lpol 2.98 / lval 0.663` (53k samples). With stopgap-vs-stopgap teacher data added it reached `lpol 2.85 / lval 0.646`, and az_v2 scored 28% vs stopgap (az_v1 10%). Source: `levels/level4/agents/az/log.txt`. z labels are per team (`1 if t == w`), so that part is correct. Source: `agents/az/sp.py:64`.
- Team constraint (coordinator): 20 CPU cores shared with three live training agents. Self-play throughput is bounded by the CPU share, not the GPU.

## Ranked recommendations (expected strength gain per hour of team time)

### Takeaway
The cheapest, safest gains come from (1) never deploying an ungated model, (2) warm-starting the net so that at step 0 it **equals hand_v1** (a residual on top of the hand evaluator with a zero-initialised head), and then (3) one or two Expert-Iteration rounds on games played by the *strongest* agent available, using soft visit-distribution policy targets and a z/Q-mixed value target with 8x symmetry augmentation. Gumbel targets and KataGo tricks are second-order in a 1-3 h window. Full AlphaZero from a weak bootstrap is the worst use of time.

### Cited Findings
- ExIt: imitating the tree-policy target (visit distribution n(s,a)/n(s)) instead of the chosen move gives "50 ± 13 Elo" more strength at near-identical top-1 error (47.0% vs 47.7%). DAgger-style extra batches sampled with the apprentice add "120 Elo". Online ExIt "substantially outperforms the batch mode". — [Anthony et al. 2017, ExIt](https://arxiv.org/abs/1705.08439)
- KataGo ablation factors (compute needed without each technique): playout cap randomization 1.37x, forced playouts + policy target pruning 1.25x, global pooling 1.60x, auxiliary policy (opponent move) target 1.30x, ownership/score aux targets 1.65x, game-specific features 1.55x; product ≈ 9.1x. — [Wu 2019, KataGo](https://arxiv.org/html/1902.10565v5)
- Using cheap search enhancements (rollouts/RAVE) only during the *early* self-play iterations fixes AlphaZero's cold start, and most such warm starts beat the baseline on three small board games. — [Wang, Preuss, Plaat 2020, Warm-Start AlphaZero](https://arxiv.org/abs/2004.12357); adaptive switch length in the follow-up [Adaptive Warm-Start MCTS](https://arxiv.org/abs/2105.06136)

### Inferences
Ranked by expected strength per hour for this team right now:
1. **Gate every candidate (cost ~5 min, avoids the biggest loss).** Games cost ~1 s CPU, so 400 alternating-colour games vs stopgap and vs hand_v1 is about 400 CPU-s, or under 2 min on 4 cores. At 60 games the 95% CI is about ±12 pp (team log shows [5%,20%] at 6/60), which cannot separate 50% from 58%. Promote only if the Wilson lower bound is above 50% vs the current deployed model.
2. **"Residual-on-hand" warm start (about 30-45 min, high expected gain, near-zero regression risk).** hand_v1 is already a 95%-vs-L3 ONNX graph (feature extractor → linear → tanh, plus hand prior). Build the new ONNX as `value = tanh(w_hand·f + g(trunk(x), f))` and `logits = hand_prior_logits + h(trunk(x))`, with the last layers of g and h **zero-initialised**, so the untrained model plays exactly like hand_v1. Then train g and h on ExIt targets (step 3). This turns the az_v1 failure mode (a fresh net worse than the evaluator it replaces) into a guaranteed starting point. Watch the 15 s CPU match budget and the 5 MiB cap: the hand feature graph runs at 1.6 ms per b8 call.
3. **ExIt round with the strongest expert (about 30-60 min per round).** Play hand_v1-vs-hand_v1 (or hand_v1 vs stopgap) through the referee-style MCTS at 100 sims and record root visit distributions (TPT targets), root Q, and z. Train, gate, and repeat with the new model as the expert. Keep all earlier data (an "exponential dataset", below). az_v2 shows the cost of weak data: adding 1000 stopgap-vs-stopgap games moved it from 10% to 28% vs stopgap.
4. **Free data multipliers (about 10 min):** 8x dihedral augmentation (see the Data section) and a z+Q value mix.
5. **Playout cap randomization for data generation (about 15 min):** more independent game outcomes per CPU-hour.
6. **Gumbel completed-Q policy targets (about 30-45 min to implement correctly):** worth it only if the policy head is not learning (lpol close to ln(#legal), see below) or if data generation drops to ≤32 sims.
7. **Aux heads (opponent's next action, end-of-game material) and global pooling in the trunk:** cheap to add if writing a new net anyway; drop the aux heads at export.
8. **Do not** run more pure AlphaZero generations from az_v1/az_v2 weights: the starting point is below stopgap and each generation inherits that.

### Gaps
- No public source measures "residual on a hand evaluator with zero-init head" specifically for AlphaZero. This is an engineering inference (the same principle as zero-init residual branches).
- The gain per hour of each step for this specific game is unmeasured. The ranking is judgement.

## Expert Iteration vs AlphaZero vs supervised bootstrapping; warm-starting without regression

### Takeaway
From a decent heuristic, supervised distillation followed by ExIt with soft tree-policy targets and the strongest available expert converges fastest. AlphaZero self-play from a net that is weaker than the heuristic regresses because the first generations learn from their own weak play. Warm-start by distilling the heuristic first (or by wrapping it), feed early search with the heuristic, and gate.

### Cited Findings
- ExIt splits the work: "Planning new policies is performed by tree search, while a deep neural network generalises those plans". It "outperforms REINFORCE" on Hex, and the final agent, trained tabula rasa, beats MoHex 1.0. — [ExIt, arXiv 1705.08439](https://arxiv.org/abs/1705.08439)
- ExIt's experts used 10,000 MCTS iterations per move. The apprentice prior weight in the expert's tree was tuned to w_a = 100, "close to the average number of simulations per action at the root". — [ExIt PDF](https://arxiv.org/pdf/1705.08439)
- ExIt sampled **one position per game** for value-network training "to ensure that there were no correlations in the dataset", used KL (cross-entropy) losses for policy and value, used early stopping instead of L2, and **reinitialised network weights each iteration** before retraining on all data. — [ExIt PDF](https://arxiv.org/pdf/1705.08439)
- ExIt online variants: (a) 24,300 new moves per iteration with a buffer of the latest 243,000; (b) keep all data and grow it by 10% per iteration. (b) was "marginally stronger, suggesting that retaining a larger dataset is useful". — [ExIt PDF](https://arxiv.org/pdf/1705.08439)
- The final DAgger TPT apprentice roughly matched its MCTS teacher (87/162 wins). — [ExIt PDF](https://arxiv.org/pdf/1705.08439)
- Warm-start AlphaZero: using rollout/RAVE-type enhancements only in the early iterations improves on the cold-start baseline in small games, and RAVE variants were strongest. — [Wang et al. 2020](https://arxiv.org/abs/2004.12357)

### Inferences
- **Why az_v1/az_v2 regressed (diagnosis from the team log):** `lpol 2.98` is close to ln(20) ≈ 3.0, meaning the learned prior is near uniform over the ~12-30 legal moves, and `lval 0.66` is barely below the variance of z under a colour-only predictor (≈0.75 when Cyan wins 25%). The net learned little beyond "which colour am I". Inside PUCT-100, a flat prior plus a weak value is weaker than hand_v1's exact threat features. Likely causes: little data from weak bootstrap players (1500 games), visit targets from a weak prior being themselves flat, and training from scratch rather than from the heuristic.
- **Warm-start recipe (pick one):**
  - (A) Residual-on-hand with zero init (see ranking item 2). Best regression safety.
  - (B) Distill first: label 200k-500k positions from hand_v1 and stopgap games with hand_v1's value (soft regression target, MSE on tanh output or BCE on (v+1)/2) and its policy prior (soft cross-entropy, masked to legal moves). Then fine-tune on ExIt targets with the value target = mix of hand value, root Q and z, annealing the hand weight from 0.5 toward 0 over the generations (the Wang et al. "warm start then switch off" idea).
- **Expert choice:** the expert should be MCTS-100 around the *current best* model (hand_v1 now). A 1-2 turn alpha-beta with the hand evaluator, like the team's `agents/distill/teacher.py`, is another valid ExIt expert. Its root move scores can be softmaxed into a soft policy target (temperature tuned so the entropy is about 1-1.5 nats), and its search value used as the value target (TD-Leaf-style bootstrapping).
- **Iteration size:** 5k-10k games per round is affordable (about 35 positions per game gives 175k-350k samples). This is 3-6x az_v1's whole dataset and matches ExIt's ~24k-243k scale.
- **Reinitialise or fine-tune:** with under 1 h per round, fine-tune from the previous weights (with zero-init residuals this is natural). Reinitialising pays off only if you also retrain on all accumulated data.

### Gaps
- I couldn't find a published head-to-head of ExIt vs AlphaZero convergence speed from a *heuristic* start. ExIt's paper compares against REINFORCE and batch variants, not AlphaZero.
- I didn't fetch numeric results from the warm-start paper (only the abstract-level claim).

## Gumbel AlphaZero/MuZero with few simulations; completed-Q policy targets

### Takeaway
Gumbel's main benefit for this team is the **policy target**: `softmax(prior_logits + sigma(completedQ))` uses the Q of every visited action and fills unvisited actions with a mixed value. It stays a valid improvement target at 16-32 sims, where visit-count targets are noisy and dominated by the prior. It's useful only for *data generation*: the referee still plays PUCT-100 with the exported net.

### Cited Findings
- Gumbel MuZero (mctx reference implementation): at the root, actions are selected by **Sequential Halving with Gumbel**. Interior nodes use a deterministic action selection. Defaults are `max_num_considered_actions=16` and `gumbel_scale=1.0`, and the docs note that "evaluation on perfect-information games can use gumbel_scale=0.0". — [mctx policies.py](https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/policies.py)
- Policy training target (`action_weights`) = softmax over legal actions of `prior_logits + completed_qvalues`. The final action is argmax of `gumbel + logits + q` over the considered actions. — [mctx policies.py](https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/policies.py)
- Completed Q (`qtransform_completed_by_mix_value`), defaults `value_scale=0.1`, `maxvisit_init=50`, `rescale_values=True`, `use_mixed_value=True`:
  - unvisited actions get the mixed value `(raw_value + N * sum_visited(p_a q_a) / sum_visited(p_a)) / (N + 1)`;
  - Q is min-max rescaled to [0,1] across actions;
  - the output is `(maxvisit_init + max_a N_a) * value_scale * completed_q`. — [mctx qtransforms.py](https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/qtransforms.py)

### Inferences
- With mctx defaults and about 16 sims (max N ≈ 6-8), sigma ≈ (50 + 7) * 0.1 ≈ 5.7 times the rescaled Q. The best-Q action gets up to about +5.7 nats over the worst, which gives a sharp, Q-driven target even when visit counts are tiny. This directly fixes the "flat target → flat prior" failure seen in az_v1 (lpol ≈ ln #legal).
- Suggested settings for this game: 16-32 sims for data generation, m = min(8, #legal) considered actions (legal moves are only 12-30), gumbel_scale 1.0 during data generation, c_visit 50, c_scale 0.1. Value is from the side to move, and the sign flips **only when the team changes** (two actions per turn), so implement this carefully.
- Interaction with the referee: a net trained on Gumbel targets has sharper priors than one trained on visit counts. Under PUCT-100 with c_puct 1.5 this concentrates visits, which is good if the targets are right. Validate by gating with the real referee MCTS, not the Gumbel search.
- Implementation cost: Sequential Halving plus completed Q is about 150 lines of Python on top of the engine. If data generation stays at 100 sims with PUCT, a cheaper partial step is to apply the completed-Q target formula to the referee-style root statistics (W/N per child plus the network's value for unvisited children).

### Gaps
- I could not open the Gumbel paper itself (OpenReview served a bot-check page): <https://openreview.net/forum?id=bERaNdoegnO>. I don't cite the paper's exact claims (e.g., "beats AlphaZero with only 2-16 simulations on 9x9 Go / chess"). They come from my memory of it and are unverified here.
- I found no evidence on Gumbel targets used for a net that is then deployed under plain PUCT.

## KataGo efficiency tricks: which matter for a small game in a short run

### Takeaway
For a 1-3 h run, use **policy target pruning** (or simply exclude noise-forced visits) and **playout cap randomization**. Add **global pooling** and **one cheap auxiliary head** if you build a new net. Ownership/score heads don't transfer directly. Game-specific input features matter a lot (1.55x in KataGo), which supports keeping hand_v1's features.

### Cited Findings
- Playout cap randomization: with p = 0.25 do a full search (N = 600, later 1000), otherwise a fast search (n = 100, later 200). **Only full-search turns are recorded** as policy targets, and fast searches disable Dirichlet noise and other exploration. Removing it costs 1.37x. — [KataGo](https://arxiv.org/html/1902.10565v5)
- Forced playouts: n_forced(c) = (k · P(c) · Σ N)^(1/2) with k = 2. Policy target pruning subtracts up to n_forced visits from each non-best child as long as PUCT(c) stays below PUCT(c*), and prunes children reduced to 1 playout. Removing both costs 1.25x. — [KataGo](https://arxiv.org/html/1902.10565v5)
- Aux targets: opponent-policy loss weight 0.15 (1.30x); ownership weight 1.5/b² and score targets (1.65x combined); global pooling (1.60x). The weights were "NOT carefully tuned". — [KataGo](https://arxiv.org/html/1902.10565v5)
- Optimisation: batch 256, momentum 0.9, per-sample LR 6e-5 (2e-5 for the first 5M samples). The window grows from 250k samples by N_window = c(1 + β((N_total/c)^α − 1)/α), with α = 0.75 and β = 0.4. — [KataGo](https://arxiv.org/html/1902.10565v5)
- KataGo overall: "50x reduction in computation over comparable methods", surpassing ELF in 19 days on fewer than 30 GPUs (out of scope for us). — [KataGo abstract](https://arxiv.org/abs/1902.10565)

### Inferences
- **PCR for this game:** full search 100 sims (same as the referee) with p = 0.25, otherwise fast 20-25 sims with no noise and no recording. CPU per game falls to about 0.25·100 + 0.75·25 ≈ 44 sims per move, about 2.3x more games, and so about 2.3x more independent z labels per CPU-hour. Policy samples per CPU-hour fall to about 0.57x. Use it when value is the bottleneck (it is, per lval 0.65).
- **Noise-forced visits:** the referee adds Dirichlet noise only in the first 8 actions. If the self-play generator copies the referee, either drop the policy target for those 8 actions or prune visits like KataGo. Otherwise the net learns to imitate noise in the opening.
- **Global pooling:** a mean/max-pooled channel vector added back to the trunk (and fed to the value head) is cheap on 8x8. az.py's value head already uses `h.mean`.
- **Aux heads that fit this game:** opponent's next action (soft target from the next position's visit distribution, weight about 0.15), "kernel captured within the next k actions" (BCE), and end-of-game material difference (MSE, small weight). Drop them at ONNX export.

### Gaps
- KataGo's factors were measured on 19x19 Go over about 2-day runs. I found no ablation for small boards or short runs.

## Value targets: z vs search Q vs a mix; TD(lambda); colour asymmetry

### Takeaway
Use a **mixed target**, e.g. `v* = 0.5·z + 0.5·q_root` (or more weight on q early in the game and on z near the end). Pure z from about 35-action games with near-greedy play is noisy and highly correlated within a game. Several studies find softened or bootstrapped targets train faster on Connect-Four-scale games. Handle the colour asymmetry through plane 15 and per-colour calibration checks, not through reweighting.

### Cited Findings
- Willemsen, Baier & Kaisers define a family of value targets (original z, soft-Z, A0C, new A0GB). soft-Z, A0C and A0GB "train faster and perform better" than the original z target on Connect-Four and Breakthrough. (I saw a search summary only, not the full text.) — [Springer, Neural Computing & Applications 2021](https://link.springer.com/content/pdf/10.1007/s00521-021-05928-5.pdf)
- Go-Exploit argues that one full self-play game yields a single noisy value target. It uses shorter trajectories started from archived states, which gives more independent value targets and better sample efficiency on Connect Four. — [Trudeau & Bowling 2023](https://arxiv.org/pdf/2302.12359)
- Lc0 discussion of the Oracle "(Q+z)/2" target: one contributor argues that training on search values removes the value head's independence. Another proposes a per-move blend (50/50 at the opening, toward z late). A third criticises Oracle's analysis because AlphaZero used temperature until the end. — [lc0 issue #84](https://github.com/leelachesszero/lc0/issues/84)
- The Oracle Connect Four series reports that about 30-50% of positions in the training window are duplicates, which in effect makes the net average their targets. (Search snippet only; the page returned 403.) — [Oracle "Lessons from AlphaZero" Part 4](https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628)
- ExIt used one position per game for value training to avoid correlated samples. — [ExIt PDF](https://arxiv.org/pdf/1705.08439)
- KataGo's game-outcome loss weight was c_value = 1.5. — [KataGo](https://arxiv.org/html/1902.10565v5)

### Inferences
- **Recommended target:** `v* = (1-a_t)·q_root + a_t·z` with a_t rising linearly from 0.3 at action 0 to 1.0 at the last action (the per-move blend from the lc0 thread). The criticism of (Q+z)/2 applies less here because referee play is greedy after action 8. If a hand value is available (warm start B), use `v* = 0.3·hand + 0.35·q + 0.35·z` for the first round.
- **TD(lambda):** with ~35 actions, lambda ≈ 0.8-0.9 over the sequence of root Q values (sign-flipped only at team changes) is a reasonable alternative to the linear blend. I found no source measuring it for AlphaZero-style training, so prefer the simpler blend under time pressure.
- **Correlation:** don't take one position per game (too wasteful at 1 s/game). Do shuffle across games and keep epochs low (≤2-3 passes per new datum) so the value head doesn't memorise trajectories.
- **Colour asymmetry:** Cyan wins about 38% (25% in the az bootstrap data, 46% in stopgap self-play), so the calibrated start value for Cyan is around 2·0.38−1 ≈ −0.24. Plane 15 lets the net learn this offset. Check the mean predicted value by colour against the empirical win rate after each training run.
- **Referee-specific pitfall (FPU = 0):** the referee scores an unvisited child at Q = 0. In positions where the side to move is losing (Cyan values around −0.2 to −0.5), every visited child looks worse than an unvisited one, so search spreads visits broadly and the prior matters less. In winning positions (often Magenta), search locks onto the first few prior-favoured moves, so **prior quality matters most for the side that is ahead**. Overconfident values (|v| near 1) amplify both effects. Keep the value head calibrated (MSE or BCE, no extra sharpening) and check search behaviour on a few positions per colour.

### Gaps
- I could not read the Oracle Part 4 numbers directly (403). The team should treat the "(Q+z)/2 helps" claim as unverified.
- I found no source on handling a built-in tiebreak asymmetry like "Magenta wins ties". The above is inference.

## Data: symmetries, replay window, samples per generation, epochs, LR, network size

### Takeaway
All piece moves are direction-symmetric, so the **full 8-element dihedral group** preserves the rules. The **anti-diagonal reflection (r,c)→(7−c,7−r)** is the one that keeps both kernels on their start corners (in-distribution). The team currently uses only the left-right flip. Keep a growing window (all data from good experts), train about 2 passes per new sample, and use a small ResNet (6x64) whose size the 5 MiB / 15 s CPU limits allow.

### Cited Findings
- Movement rules are orthogonal/diagonal/L-jump with no forward direction (Ping orthogonal 2 / diagonal capture, Daemon orthogonal slide, Exploit 2+1 L, Trace orthogonal step / diagonal shot, Ice any direction 2, Kernel king). Firewall rings are concentric. — local `levels/level4/data/colony-one-env/docs/rules.md` (Pieces, Firewall sections)
- The canonical view rotates the board 180 degrees for Magenta, so the side to move always has its kernel near (7,0). — local `docs/observation.md`; start squares from local `level-4_scene.json`
- The team's az trainer applies only a left-right flip (`FLIP` table, (r,c)→(r,7−c)). — local `levels/level4/agents/az/az.py:88-148`
- KataGo's window starts at 250k samples and grows sublinearly (α = 0.75, β = 0.4). — [KataGo](https://arxiv.org/html/1902.10565v5)
- ExIt: retaining all data with 10% growth per iteration was marginally better than a fixed buffer of ten iterations. — [ExIt PDF](https://arxiv.org/pdf/1705.08439)
- Submission cap 5 MiB, match budget 15 s CPU. — local `docs/submission.md`

### Inferences
- **Augmentation:** apply a random element of D4 to each sample: the board planes, both from/to squares of the 4096 policy index, and the legal mask. Planes 13-15 are spatially constant and plane 12 (rings) is symmetric. Tiebreak rule 2 (smaller ring wins) is symmetric. Bodyguard order is by piece type, not by square, so it is unaffected. Caveats:
  - (r,c)→(7−c,7−r) keeps the own kernel at (7,0) and the enemy kernel at (0,7), so it's the most in-distribution transform. It's safe to use always, as 2x data.
  - Transforms that move the kernels to other corners create legal but off-distribution positions. They still help generalisation for 8x8 tactics and are cheap.
  - Verify once with the engine (by a teammate, 1 core) that `legal_mask` commutes with the transform on about 100 random positions before using all 8.
- **Window:** keep everything from the hand_v1/stopgap/ExIt experts. Drop az_v1 bootstrap games from weak players. Cap at about 1-2M samples, which fits easily in RAM as uint8 planes.
- **Per generation:** 5k-10k games (175k-350k positions). Before 8x augmentation that is about 0.5-1 h of 5 cores at ~1 s CPU per game.
- **Training:** AdamW, LR 1e-3 with cosine decay to 1e-4 (or OneCycle via `ccc_2026.nn.fit`-style schedules), weight decay 1e-4, batch 512-1024, bf16 on the 5070 Ti. Do about 2 passes over new data per generation (KataGo-style sample reuse is low). Policy loss is soft cross-entropy over legal moves only, with illegal logits masked to −inf in the loss. Value loss is MSE (or BCE on (v+1)/2) with weight 1-1.5 (KataGo 1.5). GPU training of a 6x64 net on 300k samples x 2 epochs takes a few minutes, so the GPU is never the bottleneck.
- **Network size:** 6 residual blocks x 64 channels is about 0.44M trunk params (each 3x3 64→64 conv is 36.9k params). With a factorised from/to policy head (as in az.py, no 4096-wide dense layer) the total stays well under 1M params ≈ 4 MiB fp32. Check CPU latency: the referee makes about 13 calls of batch 8 per action at 100 sims, about 230 per game for one side, so even 10 ms per call is about 2.3 s. That is well inside 15 s, but measure with `tools/validate_submission.py`. If combined with hand features (warm start A), the hand graph adds about 1.6 ms per call.

### Gaps
- I haven't verified that the engine's legal-move generation is exactly invariant under transposition-type symmetries (only from the docs). It needs a quick check.
- I found no source on the best window for 1-3 h runs. ExIt and KataGo give only coarse anchors.

## Speeding up self-play with a slow Python engine and shared CPUs

### Takeaway
The engine, not the GPU, is the bottleneck, and only a share of the 20 cores is available. Use 4-5 single-threaded worker processes that each run **many games concurrently** and batch leaf evaluations across games, with fewer sims for unrecorded moves (PCR). Keep ONNX/torch inference single-threaded per worker. Prefer re-labelling existing positions (distillation) over generating new games when possible.

### Cited Findings
- PCR records only the 25% of moves that get full searches and uses cheap searches elsewhere. — [KataGo](https://arxiv.org/html/1902.10565v5)
- ExIt sampled positions with the *apprentice* (fast network play) and labelled them with the expert, noting "selecting positions with the apprentice is faster". — [ExIt PDF](https://arxiv.org/pdf/1705.08439)
- The referee search already batches 8 leaves per call with virtual loss 1.0. — local `colony_one/mcts.py`

### Inferences
- **Throughput budget:** about 1 s CPU per game at 100 sims means about 3,600 games per core-hour. With 4 cores free, that's about 14k games/hour (≈500k positions/hour). This is plenty for 1-2 ExIt rounds, so don't over-engineer.
- **Batching:** per worker, run 16-64 games in lockstep. Collect each game's pending leaf batch, concatenate them into one ONNX/torch call (CPU onnxruntime with `intra_op_num_threads=1`, or one GPU server process fed through queues). This cuts Python-to-runtime overhead per evaluation. The engine's Python move generation remains the floor.
- **Fewer sims:** 32-50 sims with Gumbel targets, or PCR with 100/25, roughly halves CPU per recorded sample.
- **Cheap teacher instead of MCTS self-play:** re-label positions already on disk (`agents/*/data`) with a stronger evaluator, e.g. a 1-2 turn alpha-beta with the hand evaluator, or MCTS-200 around hand_v1. This gives value and policy targets without new games. It's the DAgger-style shortcut, best for the value head.
- **Pin threads:** set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `torch.set_num_threads(1)` in workers, or they will oversubscribe the shared 20 cores and slow the other agents.

### Gaps
- I didn't benchmark the batched multi-game speedup for this engine. The gain depends on how much of the ~1 s is Python rules vs inference overhead, which is unmeasured.

## What top teams did in time-limited bot competitions

### Takeaway
I found little reliable public evidence in this session. The available signals point the same way: in short competitions, strong entries combine domain knowledge (rules/heuristics) with imitation of strong play and only then RL. Search plus a learned or hand evaluator beats pure end-to-end RL on short budgets.

### Cited Findings
- Kaggle Hungry Geese (2021) was won by teams from DeNA and "quantum", with 875 teams (1039 people). I didn't find the writeup itself. — [AIsmiley](https://aismiley.co.jp/ai_news/kaggle-hungry-geese-dena-quantum/)
- The Lux AI Season 2 (NeurIPS 2023) organisers describe strong rule-based, RL and imitation-learning baselines as entry points. — [NeurIPS 2023 Lux AI S2](https://neurips.cc/virtual/2023/competition/66593)
- The CodinGame "CGZero" (Oware) AlphaZero-like write-up differs from AlphaZero, including a weighted z/Q value blend (search-snippet level; the page didn't render). — [CodinGame playground](https://www.codingame.com/playgrounds/58137/alphazero-like-implementation-for-oware-abapa-game-codingame/differences-between-cgzero-and-alphazero)
- KataGo found game-specific input features worth 1.55x compute, so domain knowledge pays even in a heavy-compute RL system. — [KataGo](https://arxiv.org/html/1902.10565v5)

### Inferences
- For a 1-3 h window, the closest analogue to winning practice is: strongest heuristic or search agent → imitation (distillation) into the submitted model → short self-improvement (ExIt) rounds gated against the current best. This matches the team's distill pipeline plus hand_v1, more than the az pipeline.

### Gaps
- I couldn't retrieve primary winning writeups (Kaggle discussion pages for Hungry Geese, Lux S1/S2 and Kore; CodinGame postmortems; Battlesnake) within the time and tool budget. Kaggle pages need JavaScript and a Medium page returned 403. Claims such as "Lux S1 winner used IMPALA + UPGO + TD(lambda)" or "Hungry Geese winners used HandyRL / AlphaZero-like search" are from memory and **unverified**, so I've left them out.
- I didn't consult the Lc0 wiki or Polygames papers in this session.
