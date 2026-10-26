# Breaking the az_res self-play plateau (CCC 2026 L4, COLONY: ONE), written 15:24 UTC

**Read this first: the brief is out of date.** The plateau has already been broken once. **az_res4f_v2** went live at 15:19.
- It scored **144/220 = 65.5% [59.0, 71.4] vs az_res_v20** in the gatekeeper's c1-identical arena, and 152/200 = 76% in the az agent's own arena ([gate/RECOMMEND.md](../../gate/RECOMMEND.md), [agents/az/RESULTS.md](../../agents/az/RESULTS.md), [DEPLOYS.md](../../DEPLOYS.md)).
- Its sibling az_res4f_v1 used the same recipe and the same base but no endgame data. It scored 100/200 = 50% (az arena) and 119/220 = 54.1% (gate).
- What changed from v1 to v2, all in data:
  - 4000 endgame-start self-play games (Kernel + at most 1 piece per side, outcome-only targets), replicated x4 to 26% of samples;
  - 900 extra v20 self-play games;
  - 600 hand_v4f self-play games.
  - The offline set grew from 133,651 to 299,345 positions ([agents/az/log.txt](../../agents/az/log.txt) 14:53-15:15).
- **So the next candidate has to beat az_res4f_v2, not v20.** Everything below is written for that.

## 1. What caused the plateau, and quick checks

### Takeaway
The best-supported cause is **value labels that cannot see the low-material endgame**. The search Q inherits the value's blind spot, and outcome labels for those positions were too rare and too noisy in full-game self-play. A data change, not more steps or a new base, is what moved the score (50% → 65.5%).

Secondary causes:
- an offline buffer dominated by stale v4-v8 games;
- a constant learning rate with no annealing;
- a measuring stick (head-to-head against one sibling) that over-states gains.

Capacity (32x3) is not proven to bind.

### Cited Findings
- **Every recipe change on the same data failed:**
  - 2x steps (v30): 53/100;
  - more Q in the value target (v31: 0.5z + 0.5Q): 46% (az) / 48.3% (gate);
  - +500 v20 games (v32): 48%;
  - the new base without new data (4f_v1): 50-54%.
  - Adding endgame-start data plus fresh games gave 65.5-76% ([RESULTS.md](../../agents/az/RESULTS.md), [RECOMMEND.md](../../gate/RECOMMEND.md)).
- **All 6 v20 live losses came from K+1 vs K+1 endgames,** ending by the bare-Kernel rule or a firewall burn. MAGENTA won 1/6 lowmat endgames and 4/4 otherwise.
  - Raw v correlates 0.79 with root800 over all decisions, but only 0.26-0.63 in 5 of the 6 lost games.
  - Kernel ring depth (centralisation) decides these endgames, and the value ignores it ([reviews/az_res_v20.md](../../reviews/az_res_v20.md) §2-4).
- **The 800-sim reference shares the bias.** "Truth" in the review is outcome plus 800-sim searches with the same models, "so the real errors are, if anything, larger" ([az_res_v20.md](../../reviews/az_res_v20.md) caveat). A Q-mixed target therefore carries the blind spot back into training, which fits v31 (more Q) being worse.
- **Endgame calibration improved with endgame data** (800 held-out random endgame starts):
  - value MSE: 0.340 / 0.320 for 4f_v2 (MAG / CYAN) vs 0.393 / 0.423 for v20 and 0.392 / 0.413 for 4f_v1;
  - "won" means: +0.81 / +0.75 for 4f_v2 vs +0.70 / +0.64 for v20.
  - These are random starts, **not** the 268-299 live-loss-type positions ([RESULTS.md](../../agents/az/RESULTS.md)).
- **Pipeline facts** ([agents/az/azlib.py](../../agents/az/azlib.py) `train`, [agents/az/retrain_fast.py](../../agents/az/retrain_fast.py)):
  - every offline retrain starts from a zero-init residual;
  - AdamW at a **constant lr 3e-4**, wd 1e-4, 3000 steps x 512 (about 5 epochs over 300k positions);
  - samples are **uniform** over the buffer, with no recency, surprise or phase weighting;
  - Q-mix is one global constant (endgame samples set q = z).
- **Two arenas disagree on 4f_v2 vs v20:** 76% (az agent, 200 games) vs 65.5% (gatekeeper, 220 games).
- **Head-to-heads are non-transitive here:** distill_v3 scored 76% vs hand_v2 but 60% vs az_res_v6 ([RECOMMEND.md](../../gate/RECOMMEND.md)).
- **Literature on why a self-improvement loop can't fix what its own search can't see.** In KataGo's main run the cyclic-group blind spot did not go away by itself. It was attacked by **seeding self-play from hand-written positions**: "since December 2022, 0.08% of self-play games" start from such positions, later plus positions from games against the adversary and cyclic positions found by online players. A May 2023 net with this training beat the original adversary in **94.1%** of games without search. New adversaries still got through (47% at 4096 visits; 65% / 27% at 4096 / 65,536 visits vs the Dec 2023 net) ([FAR AI, goattack positional adversarial training](https://goattack.far.ai/positional-adversarial-training); [Tseng et al. 2024, "Can Go AIs be adversarially robust?"](https://arxiv.org/pdf/2406.12843)).

### Inferences
- The v1 → v2 jump confounds two things: the endgame data, and the larger, fresher middlegame set (v20x + sp_v4f).
  - The live losses and the calibration gain point to the endgame data as the main lever.
  - Still, a 45 s ablation (4f recipe + v20x + sp_v4f, but `end_frac = 0`) would settle it. It costs about 3-5 min including a 120-game screen.
- **Expect the next blind spot to sit one phase earlier:** the *decision to trade into* lowmat while our Kernel is shallower (W4 in the review: 12278 a40, 13136 a19, 13365 a22).
  - sp_end only creates K + at most 1 piece per side, so it never trains that decision.
  - KataGo's experience is that fixing one family of blind spots via seeded positions works for that family and then exposes the next one.
- **The 76% vs 65.5% gap is about 2.3 SE** (my calculation: SE of the difference ≈ 0.045).
  - Treat the az agent's in-loop arena as optimistic by up to ~10 pp, and trust only the gatekeeper's `garena.py`.
  - A protocol difference (seeds, warmup handling) is more likely than chance.
- A constant LR with no decay leaves the final weights on a noisy point of the trajectory. Annealing is a nearly free test, since each retrain takes 45-100 s.
- **Capacity:** 4f_v2 measures 1.27 ms per batch-8 call against a ~3 ms budget, so a 4x32 or 3x48 trunk fits. But no evidence says the 32x3 residual underfits: no train/val split is logged. Do not spend the remaining time here.

**Quick diagnostics, in order. Each takes ≤ 5 min on 1-2 cores:**
- **D1. Run `reviews/v20_lowmat.py` on az_res4f_v2** (299 live lowmat positions). The review's criterion is sign accuracy > 0.75 **and** mean v in lost-MAG positions < -0.4.
  - If 4f_v2 already passes, the endgame lever is spent on K+1 and the next data should be K+2 / pre-trade starts.
  - If not, more and better-targeted K+1 data is still the lever.
- **D2. Hold out 5% of games by game id in `retrain_fast`** and log val lpol / lval.
  - If val ≈ train: no overfitting; labels or capacity bind.
  - If val >> train: shorten training or add data.
- **D3. Measure value MSE by phase** (opening / middlegame / lowmat by piece count) on 200 fresh 4f_v2 self-play games, to find where the residual error is concentrated now.
- **D4. Count buffer age:** about 45% of positions (133k of 299k) come from v4-v8 and hand sp_v2, i.e. from weaker players whose z reflects weaker continuations.

### Gaps
- No train/val loss split exists in the logs, so overfitting vs underfitting can't be told apart from the record.
- No ablation separates the endgame data from the extra v20x / sp_v4f middlegame games in 4f_v2's gain.
- I found no source on the cause of the az-arena vs gate discrepancy.

## 2. Plateau-breaking changes, ranked, for this pipeline

### Takeaway
Before 16:20 there is time for **one bundled candidate built from data changes:**
- endgame starts regenerated by az_res4f_v2, plus new K+2 "pre-trade" starts;
- phase-dependent Q-mix (z only in lowmat);
- LR annealing;
- optionally surprise-weighted sampling.

Each retrain is about 45-100 s, so 2-3 variants can run in parallel and be screened. Reanalyse, leagues, exact solvers and bigger trunks belong after the contest.

### Cited Findings
- **KataGo policy-surprise weighting:** "about half of the total frequency weight is assigned uniformly", and the other half is "in proportion to each move's KL divergence from the policy prior to the policy target".
  - A position is written floor(w) times, plus one more with probability frac(w).
  - "We simply sample the position more frequently, using full weight for the sample without scaling." ([KataGo methods](https://raw.githubusercontent.com/lightvector/KataGo/master/docs/KataGoMethods.md))
- **KataGo short-term value targets:** exponentially averaged future values with horizons of roughly 6 / 16 / 50 turns on 19x19, which shrink with board area. The page calls its weighting test "partial and illustrative" ([KataGo methods](https://raw.githubusercontent.com/lightvector/KataGo/master/docs/KataGoMethods.md)).
- **KataGo auxiliary soft-policy head:** T = 4, weighted 8x ([KataGo methods](https://raw.githubusercontent.com/lightvector/KataGo/master/docs/KataGoMethods.md)). This applies only to training-side heads; the referee only reads policy and value.
- **MuZero Reanalyse** reruns MCTS with the latest network on stored states, so targets can be "repeatedly applied to the same state to generate better and better training targets".
  - The Reanalyse fraction alone is "enough to learn efficiently at data budgets differing by orders of magnitude".
  - Atari median human-normalised scores: 1331.7% at a 50% fraction (2000M frames), 1006.4% at 95% (200M), 126.6% at 99.5% (20M).
  - "Linear improvements in score require exponentially more data."
  - For policy targets "no off-policy issues can arise". Value targets can be biased by stale trajectories, but n-step bootstrapping with observed rewards still beat 0-step regression (126.6% vs 115.3% median at 20M).
  - The paper gives **no** Reanalyse fraction for Go or board games, and no ablation that isolates search depth or network freshness ([Schrittwieser et al. 2021](https://ar5iv.labs.arxiv.org/html/2104.06294); [NeurIPS page](https://papers.nips.cc/paper_files/paper/2021/file/e8258e5140317ff36c7f8225a3bf9590-Paper.pdf)).
- **AlphaStar league:**
  - Main agents choose opponents by PFSP, which "adapts mixture probabilities to each opponent's win rate" so they practise most against their "most problematic" opponents.
  - Main exploiters train only against the current main agents; league exploiters target the whole league.
  - Exploiters are periodically reset ([DeepMind AlphaStar blog](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)).
  - TStarBot-X argues that "using PFSP is not sufficient" and that population diversity also matters ([TStarBot-X, arXiv 2011.13729](https://arxiv.org/pdf/2011.13729)).
  - These are multi-week, multi-agent compute budgets, and I found no small-compute evidence for leagues.
- **Positional seeding at small compute (KataGo):** 0.08% of games seeded from blind-spot positions was enough to beat the original adversary 94.1% ([FAR AI](https://goattack.far.ai/positional-adversarial-training)). The team's own 26% endgame share produced +11-26 pp vs v20 ([RECOMMEND.md](../../gate/RECOMMEND.md), [RESULTS.md](../../agents/az/RESULTS.md)).
- **Lc0 learning-rate schedules:** I found only forum anecdotes.
  - test10 followed a believed-DeepMind schedule ("change LR 2 times in total, by dividing by 10"), in contrast with earlier frequent /3 cuts.
  - One poster speculated that an LR drop would fix a stall ([talkchess](https://talkchess.com/viewtopic.php?p=767419), [talkchess](https://talkchess.com/viewtopic.php?p=754512)).
  - No primary Lc0 source quantifies the Elo of an LR drop.

### Inferences: ranked changes for this pipeline

**A. Can produce a gated candidate before 16:20** (now 15:24, so about 55 min). The retrain uses the GPU and takes 45-100 s. Data generation is CPU, about 31 ms per 100-sim search (from 1.28 s per 41-action game).

| # | Change | Concrete sketch | Cost | Expected gain (my estimate) |
|---|---|---|---|---|
| 1 | **Regenerate endgame starts with az_res4f_v2 as the player, and add K+2 "pre-trade" starts** | (a) `gen_end.py models/az_res4f_v2.onnx 4000 train_v2 5`; the current file uses an older model's games.<br>(b) Copy `sp_end.random_start` into `random_start2`: 1-2 non-Kernel pieces per side (K+2 vs K+1, K+2 vs K+2), with at least one capture available within 2 actions so the game crosses the trade decision. Kernels on varied rings, both colours to move, rounds 6-20. About 2000 games.<br>(c) 200 starts from the 6 live-loss seed positions (12278 a43, 13136 a22, 13365 a25, 11686 a36, 11791 a27, 12113 a26), each with 1-2 random piece shifts.<br>All with q = z (outcome only), at an `end_frac` of about 0.30. | Data about 3-5 min on 5 workers (K+1 games are about 4.9 positions; K+2 games longer); retrain about 60 s | The largest lever. It targets the W4 trade decision, which K+1 starts never reach. +5-10 pp vs 4f_v2 if D1 shows K+1 is fixed and the losses move to the trade |
| 2 | **Phase-dependent Q-mix: z only in lowmat positions** | In `Buffer.tensors`, compute per-sample piece counts from obs planes 0-11. Where both sides have ≤ 2 pieces including the Kernel, use `qmix = 0`; elsewhere 0.3 (or anneal Q-mix toward 0 as total material falls). One line plus a mask. | About 5 min of code plus a retrain | Removes the blind Q from exactly the positions where it is wrong. +0-5 pp; it should show up in D1 calibration first |
| 3 | **LR annealing** | `torch.optim.lr_scheduler.OneCycleLR(max_lr=3e-4, total_steps=3000, pct_start=0.1)`, or cosine to 3e-6 over the last 1000 steps. Keep 3000 steps. | 1 line; reruns free | +0-4 pp; standard practice, no primary game-RL source found. Free to bundle |
| 4 | **Surprise / hard-position weighting** (KataGo style) | Per position, w = 0.5 + 0.5 · KL(p_target ‖ softmax(prior logits of the current model)) / mean(KL). Up-weight x2 positions from lost games and positions with \|raw v − Q800\| > 0.5 (from `v20_scan_*.json`). Sample via `torch.multinomial(w, batch, replacement=True)` instead of `randint`. | About 10 min (one forward pass over 300k positions for the KL) | +0-3 pp. It is prioritized replay without importance weights, which is KataGo's choice |
| 5 | **Reanalyse-lite** | Relabel `p` and `q` for **only** the old v4-v8 + sp_v2 positions in the lowmat phase plus all positions from lost games, about 20k positions, with az_res4f_v2 at 100 sims. Do **not** relabel the whole 133k (about 69 CPU-min). | 20k x 31 ms ≈ 10 CPU-min, about 2-3 min on 5 workers | Low: Q from the same blind net does not fix the blind spot; it only freshens policy targets. Do it only if cores are idle |

- Run three retrains in parallel on the GPU:
  - **A** = 1 + 2 + 3;
  - **B** = A + 4;
  - **C** = the D-ablation, without new endgame data.
- Screen each with 120 games vs az_res4f_v2 on the gatekeeper's `garena.py`, extend the best to 300-400 games, and add the gauntlet (section 4).
- Train from scratch on the 4f base, as 4f_v2 did. Do not warm-start from 4f_v2, which avoids compounding Q bias.

**Not worth it today:**
- **KL-to-previous-policy regularisation.** The frozen base plus the zero-init residual already anchors the policy, and each retrain is offline from scratch.
- **Alpha-beta teacher games mixed into self-play.** The distill students scored 42-50% vs v20. Teacher labels have not been shown to be better in lowmat, and the gate shows the students' trade-avoidance prior (distill_v5hf) without better endgame values.
- **A larger trunk.** Unproven need; adds latency-validation risk.

**B. After the contest**, in rough order of expected value:
1. **An exact endgame tablebase for the bare-Kernel family** (Lc0-style tablebase rescoring), then rescore training outcomes and seed positions.
   - The open board after round 8 is 6x6 (36 squares), with K + at most 1 piece per side.
   - State count: 36^4 ≈ 1.7M Kernel/piece placements x 25 type pairs x side x actions_left x lock phase ≈ 10^8, which is feasible offline in numpy/C.
   - The bare-Kernel resolution (Chebyshev ≤ 2: the mover who just finished loses; else the deeper ring wins; ties go to Magenta) gives exact terminal labels.
2. **Full Reanalyse** of the buffer with a deeper search (400-800 sims) by the newest net every generation, at a Reanalyse fraction of 50-80%. MuZero's evidence is Atari, so the fraction must be tuned.
3. **PFSP opponent mix in the loop:** replace the fixed 100 games vs az_res_v6 with a pool of {v6, v20, 4f_v2, hand_v4f, distill_v5hf}, weighted toward the opponents we score worst against. Add a main-exploiter: a residual trained only to beat the current seat, whose wins become seed positions.
4. **KataGo extras:** playout-cap randomisation, short-horizon TD value heads, a larger trunk (4x32 or 3x48 within 3 ms), and an auxiliary soft-policy head.

### Gaps
- No source found for small-compute (single-machine, hour-scale) gains from leagues, PFSP or KL regularisation; all the evidence found is large-scale.
- No primary Lc0 or KataGo source with an Elo figure for LR annealing in self-play training.
- MuZero Reanalyse gives no board-game fraction and no ablation of reanalysis search depth.
- The optimal endgame mix ratio is unknown: 26% is the only tested point.

## 3. How to get correct value labels where the search is blind

### Takeaway
Use **outcome-only labels from many short restarts** of the blind-spot positions, which is what worked. Extend it to the pre-trade phase, and average several playouts per seed to cut label noise. An exact solver is the real fix but a post-contest job. Teacher labels have not been shown to beat the net here, and Q-mixed labels in these positions are actively harmful.

### Cited Findings
- **Outcome-only endgame starts moved the endgame calibration and the head-to-head; Q-mixing did not:**
  - endgame-start data gave MSE 0.34 / 0.32 vs 0.39 / 0.41;
  - 4f_v2 scored 65.5% vs v20 while 4f_v1 scored 54%;
  - more Q made it worse (v31 46-48%) ([RESULTS.md](../../agents/az/RESULTS.md), [RECOMMEND.md](../../gate/RECOMMEND.md)).
- **Endgame-start games average 4.9 positions** (19,442 positions / 4000 games, [log.txt](../../agents/az/log.txt) 15:14), so z sits a few moves from the terminal and has low variance.
- **The bare-Kernel rule is closed-form at turn end** ([reports/Similar games model improvements.md](../../reports/Similar%20games%20model%20improvements.md); rules.md:64-66 as cited there), so exact terminal labels are available for the K vs K tail.
- **Lc0 used tablebase rescoring of training data** to improve endgame play, according to a 2018 forum post. This is anecdotal; I found no primary documentation ([talkchess](https://talkchess.com/viewtopic.php?p=774171)).
- **KataGo fixed its search-blind cyclic positions by seeding games from those positions**, not by changing value targets ([FAR AI](https://goattack.far.ai/positional-adversarial-training)).

### Inferences
- **Label quality ranking for today:**
  1. z from restarts played by the current best model, with q = z;
  2. the same, averaged over k = 2-4 playouts from the identical seed (the Monte Carlo mean cuts variance by about 1/k);
  3. teacher alpha-beta values only if the teacher is first shown to beat the net's sign accuracy on the 299 lowmat positions (untested).
- **Never Q-mix in lowmat.**
- Use the strongest model as the restart player. A weaker player's outcomes encode its own endgame mistakes. `agents/az/data/games_end_train1.pkl` was written at about 15:13 UTC, before 4f_v2 was exported at 15:15. So it came from an earlier model (which one is not logged), and regenerating with 4f_v2 is a free label upgrade.

### Gaps
- No measurement of the distill alpha-beta teacher's accuracy on the lowmat positions, so its value as a label source is unknown.
- No primary source for Lc0 rescoring details.

## 4. Is "head-to-head vs v20" the right measuring stick?

### Takeaway
**No.** A head-to-head against one sibling is necessary but not sufficient. Non-transitivity is documented locally, and the in-loop arena runs about 10 pp optimistic. The minimum before an upload:
- a head-to-head vs the current seat (az_res4f_v2) in `garena.py`;
- plus a small gauntlet;
- plus the lowmat calibration as a regression suite.

### Cited Findings
- distill_v3 scored 76.7% vs hand_v2 but 60.0% vs az_res_v6; for 4f_v2 vs v20 the two arenas gave 76% and 65.5% ([RECOMMEND.md](../../gate/RECOMMEND.md), [RESULTS.md](../../agents/az/RESULTS.md)).
- The gatekeeper itself notes: "distinct is not the same as independent of the opponent pool ... Treat the vs-hand_v2 margin as an upper bound on the live Elo gain" ([RECOMMEND.md](../../gate/RECOMMEND.md)).
- The AlphaStar and TStarBot-X work treats non-transitivity as the reason for population-based evaluation and training ([DeepMind](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/), [TStarBot-X](https://arxiv.org/pdf/2011.13729)).
- KataGo's adversarially trained nets beat the original attack but lost to new adversaries, so a strength gain on one opponent does not carry over to others ([Tseng et al. 2024](https://arxiv.org/pdf/2406.12843)).

### Inferences
**Suggested gauntlet, about 8-10 min wall on 4 workers when the box is not saturated:**

| check | games | pass condition |
|---|---|---|
| vs seat az_res4f_v2 | 300-400 | see section 5 |
| vs az_res_v20 | 100 | ≥ seat's 65.5% − 10 pp |
| vs az_res_v6 | 60 | ≥ 70% |
| vs hand_v4f | 60 | ≥ seat's 82% − 10 pp |
| vs distill_v5hf | 60 | ≥ 50%; it is the stylistically different, trade-avoiding model, so a cheap non-transitivity probe |
| `v20_lowmat.py` | 299 positions | sign accuracy ≥ seat's, and MAG-lost mean v not higher than seat's |
| greedy / stopgap | as now | as now |

- Weight the head-to-head vs the seat most, because live opponents are many, unknown teams.
- A candidate that wins the head-to-head but regresses in the gauntlet signals a sibling-specific exploit, not general strength.

### Gaps
- There is no local measure of how well head-to-head margin predicts the live rating change. v6 → v20 → 4f_v2 live deltas could calibrate it once 4f_v2 has about 20-30 games.

## 5. Is the upload gate (Wilson 95% LB > 0.55 vs the seat) right for a +8 pp candidate with ~1.5 h left?

### Takeaway
**No.** At Wilson LB > 0.55 the gate is effectively a "≥ +12-15 pp" gate.
- A true 58% candidate passes only **14% of the time at 220 games** and **22% at 400**.
- Forfeits turned out to be a download artefact (DEPLOYS 15:09). So an upload costs only variance, and a real +8 pp edge against the pool is positive expected value with 30-60 games left: **+43 to +51 rating, P(worse than keeping) 0.24-0.28** in my Glicko-2 simulation.

Recommendation: **Wilson LB > 0.50 at 300-400 games vs the seat, plus the gauntlet in section 4**, before 16:00. After 16:00, keep a stricter bar (LB > 0.55), because the realised gain shrinks and the head-to-head overstates the pool edge.

### Cited Findings
- **Gate history.** The rule is ≥ 60 games, LB > 0.55, and > 0.60 after 16:00. The warm wrapper failed it with 318/560 = 56.8% [52.7, 60.8] and pooled 557/960 = 58.0% [54.9, 61.1] ([DEPLOYS.md](../../DEPLOYS.md), [RECOMMEND.md](../../gate/RECOMMEND.md)).
- **Forfeits.** "the earlier '2 swap forfeits' were a download artefact ... Each upload costs only ~10-12 games (8-15 min) of +-35-85 swing, no systematic loss" ([DEPLOYS.md](../../DEPLOYS.md), 15:09-15:12 notes).
- **Game rate.** The round-1 report gives ≈95 games/h per seat ([reports/AI vs AI game model optimization.md](../../reports/AI%20vs%20AI%20game%20model%20optimization.md)). DEPLOYS later revised this to about 0.76-0.8 games/min, i.e. about 46-48/h ([DEPLOYS.md](../../DEPLOYS.md)). From a 16:20 upload to 17:00 that leaves about 32-63 games.
- **The round-1 Glicko-2 simulation** (σ 0.10, 30 games after upload) gave:

  | true edge | mean rating change | SD | P(worse than keeping) |
  |---|---|---|---|
  | +20 pp | +106 | 51 | 0.05 |
  | +10 pp | +51 | 51 | 0.20 |
  | +5 pp | +24 | 55 | 0.34 |

  Source: [reports/AI vs AI game model optimization.md](../../reports/AI%20vs%20AI%20game%20model%20optimization.md).

- **Gate power (my calculation),** the exact binomial probability that the Wilson lower bound clears the threshold, from scratchpad `plateau_gate.py`:

  | threshold | n | p = 0.55 | p = 0.58 | p = 0.60 | p = 0.65 |
  |---|---|---|---|---|---|
  | LB > 0.55 | 120 | 0.03 | 0.10 | 0.20 | 0.62 |
  | LB > 0.55 | 220 | 0.02 | **0.14** | 0.32 | 0.86 |
  | LB > 0.55 | 400 | 0.02 | 0.22 | 0.52 | 0.98 |
  | LB > 0.50 | 220 | 0.32 | 0.67 | 0.85 | 1.00 |
  | LB > 0.50 | 300 | 0.43 | **0.81** | 0.94 | 1.00 |
  | LB > 0.50 | 400 | 0.52 | 0.90 | 0.98 | 1.00 |

- **My Glicko-2 simulation** (standard Glicko-2, τ 0.5, one game per period, start rating 2000, RD 90; σ 0.10 after an upload vs 0.06 when keeping; opponents at our rating ± N(0, 120), RD 70; 1500 runs; edge = true win-rate edge against the live pool; same script):

  | games left | edge | forfeits | mean vs keep | SD | P(worse) |
  |---|---|---|---|---|---|
  | 30 | +4 pp | 0 | +22 | 73 | 0.39 |
  | 30 | **+8 pp** | 0 | **+43** | 75 | **0.28** |
  | 30 | +8 pp | 2 | +31 | 73 | 0.34 |
  | 30 | +15 pp | 0 | +83 | 73 | 0.12 |
  | 60 | +4 pp | 0 | +25 | 72 | 0.36 |
  | 60 | **+8 pp** | 0 | **+51** | 72 | **0.24** |
  | 60 | +15 pp | 0 | +101 | 71 | 0.08 |

### Inferences
- **The 0.55 bar was set under the assumption of about 2 forfeits per upload.** Without forfeits it rejects most real +8 pp candidates, about 86% at 220 games.
- **LB > 0.50 at 300 games** passes a true 58% candidate 81% of the time. It also passes a true 55% candidate 43% of the time, which is still positive expected value (+4 pp: +22 to +25, P(worse) 0.36-0.39).
- **The real risk is a sign error from non-transitivity, not a small edge.** Hence the gauntlet requirement in section 4.
- **Head-to-head edge ≠ pool edge.** Sibling head-to-heads have over-stated the live gain (76% vs 65.5% between arenas; distill_v3 76% → 60%). So a measured +8 pp vs 4f_v2 may be +4-8 pp against the pool, which is still positive expected value per the table.
- **Timing:**
  - A +8 pp candidate should be uploaded **by about 16:00** to get the ~60-game column. By 16:20 the 30-game column applies (+43 mean, P(worse) 0.28).
  - After 16:20, only a ≥ +15 pp candidate is worth uploading (P(worse) ≤ 0.12).
  - The DEPLOYS spacing rule also applies: wait about 20 games after an upload, so the next slot after 4f_v2 (live 15:19) opens about 15:40-15:45.
- **A rank-based objective changes the answer.** If the team only cares about holding its rank (not expected rating), the SD of about 72-75 per upload matters, and a stricter gate is defensible. The team should decide which objective it optimises.
- **To pass the current gate, run more games, not fewer.** If the 0.55 bar is kept, a real 58% candidate needs about 1000+ games to pass reliably, which is not available. Choose the threshold by objective instead.

### Gaps
- The server's actual post-upload σ, τ and RD reset are unpublished, so the SD figures depend on assumptions. The means are robust to σ per round 1.
- The true game rate conflicts between the round-1 report (95/h) and DEPLOYS (46-48/h). I simulated both 30 and 60 remaining games for that reason.
- The mapping from local head-to-head edge to live pool edge is unmeasured.
