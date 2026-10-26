# Referee-MCTS levers: tuning a policy/value model for a fixed PUCT search (COLONY: ONE, L4)

Sources marked "local" are files in this repo. "Replica" means my own pure-NumPy copy of the referee's
selection, virtual-loss and batching loop (`_select_action`, the batch loop in `run`), run on one CPU
core. It uses synthetic root children (each child returns a fixed value q_i ± noise on every visit).
The script is [root_bandit.py](file:///C:/Users/melis/AppData/Local/Temp/claude/C--Users-melis-Downloads-cloudflight-coding-contest/2537b5b1-8fcc-4ac8-8fd3-b627f7b7b210/scratchpad/root_bandit.py)
plus [root_bandit_b2.py](file:///C:/Users/melis/AppData/Local/Temp/claude/C--Users-melis-Downloads-cloudflight-coding-contest/2537b5b1-8fcc-4ac8-8fd3-b627f7b7b210/scratchpad/root_bandit_b2.py).
It models only the root. Depth and real game trees are not modelled.

## Answer for main session: value and prior scaling

### Takeaway
Under this exact search, **the prior is the dominant lever, and the value should be de-saturated**.
The details:
- Keep non-decisive value pre-activations at |z| ≲ 1 (about stopgap's `mat`=1/80–1/100 per material
  point, i.e. v_scale ≈ 0.4–0.6 on the "soft" terms). Leave decisive terms (kernel in fire, kernel
  capturable now) fully saturated.
- Give the prior a **2–3 logit lead for good captures and ≥5 for kernel captures**.
- Add small (0.1–1 logit) tie-breaks among quiet moves.
- Never let a must-consider move (kernel escape, guard move, recapture) fall outside the **top ~5
  priors**.

The team's own sweep agrees on direction: flat prior 27% (much worse), v_scale 2 32% (worse),
v_scale 0.5 48% and p_scale 2 50% (both neutral within ±12 pp).

### Cited Findings
**What the code actually does** (all from local [mcts.py](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/mcts.py)):
- **FPU is absolute 0.** Selection uses `q = (W - virtual_loss*vl)/n if n > 0 else 0.0` and
  `u = c_puct*p*sqrt(total+1)/(1+n)`. Here `total` counts real visits plus pending virtual-loss
  visits (lines 342–356). An unvisited edge always has Q = 0, whatever the parent's value.
  — [mcts.py L342-356](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/mcts.py)
- **Ties go to the first legal action in dict order.** The comparison is strict (`if score >
  best_score`), and priors are inserted in legal-action order (lines 308–311, 357–359). With equal
  priors, move-generation order decides which quiet moves get searched first.
  — [mcts.py L308-311, L357-359](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/mcts.py)
- **Backup within a turn.** Every path entry stores `(node, action, node_sign)`, and backup does
  `W += value * node_sign * value_sign` (lines 546–548). After the first action of a turn
  (actions_left 2→1) the child keeps the same team and the same sign (lines 431–438). The sign flips
  only at `_turn_end` (lines 521–522).
  - So the model's value at an actions_left=1 position (plane 13 = 0.5) is added *unchanged* to the
    root edge of the same team's first action.
  - The value at the opponent's fresh-turn position (actions_left=2) is added *negated*.
  - A root first-action edge's Q is therefore a mix: its first visit is the model's al=1 value; later
    visits are negated opponent al=2 values and deeper values.
  — [mcts.py L431-438, L521-522, L546-548](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/mcts.py)
- **The referee's own docstring says first actions are under-searched.** "At actions_left == 2 the
  same simulation lands in an unfinished position ... a root child must be visited several times
  before its own second action is searched at all. Without this factor, first actions would be
  systematically under-searched". The factor it refers to (`adaptive_simulations`) is **off** by
  default (line 195), and the referee changes only `simulations`.
  — [mcts.py L195, L659-666](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/mcts.py);
  [submission.md](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/docs/submission.md)
- **Kernel captures are terminal.** A kernel capture returns +1.0 at once (lines 413–418). It gets no
  evaluator call and no virtual loss, and is backed up immediately (line 626). So once a kernel
  capture is found, later descents in the same batch can pile visits onto it.
  — [mcts.py L413-418, L624-627](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/mcts.py)
- **The search is called once per action.** The second action of a turn gets a fresh tree. Level 4
  uses 100 simulations, c_puct 1.5, batch 8 and VL 1.0.
  — [submission.md](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/docs/submission.md)

**stopgap's actual scales** (local [stopgap.py](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/stopgap.py)):
- **Value pre-activation** z = (own−enemy material)/40 + kernel/positional terms. VALUES are P15 D45
  X25 T30 I50 K100. That gives:
  - Ping down: z = −0.375 → v = −0.36
  - Daemon down: z = −1.125 → v = −0.81
  - Ice down: z = −1.25 → v = −0.85
  - Daemon + Ice up: z = 2.375 → v = 0.98
- **Positional terms** are small: center ≤ 0.25, guard 0.35, threat 0.25. The hang term is
  0.3 × value/40 (Daemon attacked → 0.34). The fire-danger term is 1.5 × urgency. Plane 13
  (actions_left) is **not read**, so al=1 and al=2 positions get the same value function.
  — [stopgap.py](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/stopgap.py)
- **Prior.** to_term = enemy value/20, giving logits Ping 0.75, Exploit 1.25, Trace 1.5, Daemon 2.25,
  Ice 2.5, Kernel 5.0. Squares about to burn get −0.5·urgency, and a kernel standing in fire gets a
  +1.5 from-bonus. **All quiet moves sit at exactly 0**, so they are tied and searched in
  move-generation order.
  - With 20 legal moves, one Daemon capture has P = e^2.25/(e^2.25+19) = 0.33, and each quiet move
    0.035. At p_scale 2 the capture's P is 0.83.
  — [stopgap.py](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/stopgap.py)

**Team arena sweep vs stopgap_v1** (60 games each, local [sweep_results.jsonl](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/sweep_results.jsonl)):

| Variant | Win rate | 95% CI | Notes |
|---|---|---|---|
| v_scale 0.5 | 48% | 36–61 | |
| v_scale 2 | 32% | 21–44 | Cyan 10%, Magenta 53% |
| p_scale 0 (flat prior) | 27% | 17–39 | |
| p_scale 2 | 50% | 38–62 | |

**Replica, test A: how much value advantage overturns a prior lead.**
- Setup: 20 moves, 100 sims, batch 8, VL 1, per-visit noise sd 0.15. Move0 is the prior favourite
  (logit L, value q). Move1 is truly better (logit 0, value q+g). The other 18 moves are q−0.1.
- Each cell is the share of 200 searches that end up playing move1. Base value 0.0 is shown; base
  −0.4 and +0.4 were nearly identical.

| L (P0) | g=0.05 | 0.10 | 0.20 | 0.40 |
|---|---|---|---|---|
| 0 (0.05) | 0.56 | 0.81 | 0.98 | 1.00 |
| 1 (0.13) | 0.13 | 0.34 | 0.82 | 1.00 |
| 2 (0.28) | 0.00 | 0.00 | 0.18 | 0.99 |
| 3 (0.51) | 0.00 | 0.00 | 0.00 | 0.73 |
| 4 (0.74) | 0.00 | 0.00 | 0.00 | 0.14 |
| 6 (0.96) | 0.00 | 0.00 | 0.00 | 0.01 |

— replica (root_bandit.py, test A)

**Replica, test B2: discovery of a winning low-prior move** (q = +0.9, logit 0, the other 18 moves at
logit 0 with value = base). Visits to the hidden move in 100 sims:
- **Hidden move LAST in tie order:**
  - base −0.5: found at every L (58–71 visits)
  - base 0.0: found up to L=4; **0 visits at L=6**
  - base +0.3: 14 visits at L=1, 7 at L=2, **0 for L≥3**
  - base +0.6: **0 even at L=1**
- **Hidden move FIRST in tie order:** found in every case (21–80 visits). The first batch's virtual
  loss pushes the 2nd–8th prior-ranked moves into one free 1-ply evaluation.
- **Hidden move given a +0.3 logit tie-break (test B3):** found in every case (21–74 visits).
— replica (root_bandit_b2.py)

**Replica, test C: value scale vs a stopgap-like prior.**
- Setup: prior favourite = Daemon-capture logit 2.25 with q 0.10. The truly best move is an
  Exploit-capture logit 1.25 with q 0.25. Values are rescaled by k in the pre-activation.
- Share of searches picking the better move:

| k | Picks better move |
|---|---|
| 0.25 | 0% |
| 0.5 | 0% |
| 1 | 26% |
| 2 | 81% |

— replica (root_bandit.py, test C)

### Inferences
**(a) Value: ideal pre-activation and calibration.**
- **Value scale works like 1/c_puct.** Scaling all values by k gives the same argmax as dividing
  c_puct by k: Q+U ∝ Q + (c/k)·U′.
  - Absolute FPU = 0 is scale-invariant, but VL 1.0 and terminal ±1 are not.
  - So v_scale 0.5 ≈ effective c_puct 3.0 (wider, prior-led), and v_scale 2 ≈ c_puct 0.75
    (value-greedy). The arena says value-greedy loses (32%).
  - The likely reason: a value with ~1 ply of real tactics is noisier than the capture prior. Each
    first-action root edge is judged mostly by its al=1 leaf, which ignores the opponent's 2-action
    reply.
- **tanh saturation distorts move comparisons at stopgap scale:**
  - From level material, winning a Ping moves v by 0.36. That alone overturns a 2-logit prior lead
    (test A needs g ≈ 0.2–0.4).
  - When already a Daemon up (z = 1.1), winning another Daemon moves v only 0.81 → 0.98 = 0.17.
  - So the search is hypersensitive to material in equal positions and indifferent when ahead,
    which is the reverse of what conversion needs. Also, |v| ≥ 0.95 non-terminals look almost like
    a kernel kill (+1).
- **FPU = 0 makes the value's sign an exploration switch.** Test B2 shows this:
  - When the mover's visited moves score ≥ +0.3, moves outside the top ~5–8 priors are **never
    looked at**.
  - When they score ≤ 0, everything gets tried.
  - Overconfident positive values therefore blind the winning side to hidden defences or kills.
    Overconfident negative values make the losing side's search wide and shallow.
  - Shrinking values toward 0 softens both effects.
- **Numeric rule (proposal, not measured):** z ≈ material_diff/90 + Σ positional (each ≤ 0.15) +
  ~0.6 × SEE-style hanging gains/90 + decisive terms (±3).
  - Target: typical non-decisive |v| ≤ 0.75, so one-move Q differences land in the 0.05–0.4 range
    the search can use.
  - Reserve |v| > 0.95 for near-forced outcomes: kernel unguarded and attacked with the attacker to
    move, kernel on a ring about to lock, a won bare-kernel ending.
  - In stopgap terms: v = tanh(0.45·soft + hard), where hard = the w_danger kernel-in-fire term
    (plus any "win now" term) and soft = everything else.
- **Calibration target: win probability, 2p−1.** That is what AlphaZero-style MSE value training
  converges to (E[z]).
  - A logistic fit P(win) = σ(a·z + b·magenta) maps exactly to v = tanh((a·z + b·magenta)/2),
    because tanh(x) = 2σ(2x) − 1.
  - From replayed tournament/arena games (`analyze_matches.py` already replays protocols), fit `a`
    and `b` on mid-game positions, then set v_scale = a/2 and add a ±b/2 colour term on plane 15.
    That is about 30 minutes of work.
  - Guess, unverified: a Daemon up is unlikely to be an 81% (v = 0.81 ⇒ p ≈ 0.90) win this early,
    so the fit will probably land near v_scale 0.4–0.6.

**(b) Prior: logit spread over ~12–30 legal moves.**
- **Rule from test A:** at 100 sims, a logit lead of L costs about **0.1 (L=1), 0.25 (L=2),
  0.4 (L=3), >0.5 (L≥4)** of value advantage to overturn. Pick each class's lead by how big a value
  gap you want the search to be able to overrule.
- **Recommended spreads above the quiet-move baseline:**
  - Kernel capture: ≥5. It is terminal anyway; the prior just needs it visited first.
  - Clean captures of undefended pieces: 2–3, scaled by target value (the current /20 is right).
  - Captures into a defended square: target − attacker value, /20 (SEE), floor 0.
  - Kernel escape / guard moves when the own kernel is attacked: +2 to +3.
  - Moving a piece onto an enemy-attacked, undefended square: −(piece value/50), i.e. about −1.
  - Quiet moves: a 0–1 positional spread (towards enemy kernel or centre, away from the burning
    ring, keep guards adjacent). Never exact ties.
- **The top-5 rule (tests B2/B3).** The VL in batch 1 gives roughly the top 5–8 prior-ranked moves
  one free 1-ply evaluation (8 distinct moves at P_top 0.3, 5 at P_top 0.9).
  - Below that rank, a move is examined only if the mover's visited moves score ≤ ~0.
  - So the **rank of critical moves matters more than the absolute spread**. A +0.3 tie-break was
    enough to make a winning move reliably found.
- **Applies to both sides.** The same prior runs at the opponent's nodes. A capture-first prior is
  also the search's threat detector: an enemy kernel capture at logit 5 is the first thing searched
  at every opponent node.
- **p_scale (= 1/softmax temperature) interacts with v_scale.** Test C shows that shrinking the value
  with an unchanged prior makes the search always follow the prior. Shrinking v *and* sharpening p
  (v0.5 + p2) gives an almost pure prior player. Test the combinations, not each knob alone.

**Candidate settings to A/B** (≥200 colour-balanced games each; 60 games only resolves ±12 pp):
1. **S1 "de-saturate":** v = tanh(0.5·soft + 1.0·hard) with the hard part = w_danger kernel-in-fire
   term, p_scale 1.0.
2. **S2 "de-saturate + sharper captures":** S1 with p_scale 1.5, plus +0.3 tie-break logits for
   quiet moves by ring depth / distance to enemy kernel.
3. **S3 "global shrink":** v_scale 0.5 (already about even at 60 games) with p_scale 1.5. If S1–S3
   tie, S1/S2 is the safer choice: it keeps decisive terms decisive.
4. Optional fourth: S2 plus a colour term +0.1·(2·plane15 − 1), Magenta-favourable. Better: use the
   fitted b/2 from the logistic calibration.

### Gaps
- No measured win-probability calibration of stopgap's z yet. The 1/80–1/100 material rule is an
  inference from search mechanics plus the v_scale sweep, not a fit.
- The replica models only the root with fixed child values. The real tree's deeper backups, the
  second-action re-search and noise structure are not modelled. Table thresholds are indicative,
  not exact.
- The sweep compares against stopgap_v1 itself. Gains against the tournament field may not be
  transitive.

## FPU = 0 with values in [-1,1]: effect on exploration vs exploitation, and exploitable offsets

### Takeaway
Absolute FPU 0 means "an unvisited move is assumed to be an even position".
- When the mover's evaluated moves are negative, every unvisited move looks better and the search
  goes wide.
- When they are positive (≥ +0.3), moves outside the top ~5–8 priors are never visited.

This is far more optimistic than AlphaZero (unvisited = loss) and than KataGo/Lc0 (parent − reduction)
for a side that is behind. The only offset that is consistent across tree depths is a
**colour-antisymmetric** one, and it acts as a per-colour FPU shift. A constant "side-to-move" offset
creates an odd/even depth bias and should be avoided.

### Cited Findings
- AlphaZero's actual search used values in [0,1], and unvisited nodes were initialised to the loss
  value. Matthew Lai (AlphaZero team), 7 Dec 2018: "All the values in the search are [0, 1]. We store
  them as [-1, 1] only for network training" and "Yes, all values are initialized to loss value."
  — [TalkChess](https://talkchess.com/viewtopic.php?p=781770)
- The Monte-Carlo Graph Search for AlphaZero paper treats unvisited nodes as losses (−1).
  — [arXiv 2012.11045](https://arxiv.org/pdf/2012.11045)
- KataGo scores unvisited children at the parent's value minus an FPU reduction: c_FPU = 0.2, scaled
  by the square root of the policy mass already explored, and 0 at the root when Dirichlet noise is
  on. The paper also notes a noised move's "initial evaluation might be negative, preventing further
  search and leaving the move undiscovered."
  — [KataGo paper, arXiv 1902.10565](https://arxiv.org/pdf/1902.10565)
- Lc0 offers FpuStrategy "reduction" (default) or "absolute" (FpuValue default −1.0, i.e. loss). The
  2019 options page lists an FPU reduction default of 1.20.
  — [Lc0 options wiki](https://github.com/LeelaChessZero/lc0/wiki/Lc0-options)
- A large FPU makes the search try every move once before exploiting; a small FPU lets exploitation
  start sooner, and a high FPU can make the search shallow.
  — [Batch MCTS, arXiv 2104.04278](https://arxiv.org/pdf/2104.04278) (summary via search snippet)
- Community tuning of Lc0's FPU reduction gave mixed values (−0.08, later 0.14–0.19). Forum reports,
  not official.
  — [TalkChess p=762876](https://talkchess.com/viewtopic.php?p=762876); [TalkChess p=763989](https://talkchess.com/viewtopic.php?p=763989)
- Referee: `q = ... if n > 0 else 0.0`, absolute, not parent-relative.
  — [mcts.py L354](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/mcts.py)
- Replica test B2 (hidden winning move last in tie order): found at base −0.5 for all L; never found
  at base +0.6 even with a 1-logit gap; at base +0.3, never found for L ≥ 3.
  — replica (root_bandit_b2.py)

### Inferences
- **Compared with KataGo's FPU** (parent − 0.2):
  - The referee explores **more** whenever the parent's value is below +0.2. With the parent at −0.5,
    KataGo's FPU is about −0.7 while the referee's is 0, so the referee tries every move.
  - It explores **less** when the parent is clearly winning. With the parent at +0.6, KataGo's FPU is
    about +0.4, still above the referee's 0.
  - Compared with AlphaZero (FPU = loss), the referee is always wider.
- **Effect on a game tree.** If our side is ahead (+v), our nodes narrow and the opponent's nodes
  (−v) widen: the opponent searches all its replies. That is reasonable defensive behaviour, but it
  costs depth on our principal line. When we are behind, our root goes wide, and the move choice
  becomes nearly prior-driven with ~5 visits per move.
- **Offsets, derived from the backup code (L431-438, L521-522, L546-548):**
  - A constant mover-perspective offset +c adds +c to depth-1 leaves (same team, al=1) and −c to
    depth-2/3 leaves (opponent), then +c again at depth 4/5.
  - Root Q averages leaves of mixed depth, so lightly explored moves (mostly depth-1 leaves) get +c
    relative to deeply explored ones. That is an odd/even-depth bias, so **do not add one**.
  - A colour-antisymmetric offset (+m when Magenta is to move, −m when Cyan is) is the same from each
    node's own perspective at every depth. Cyan nodes see all non-terminal Q's shifted −m (wider,
    since unvisited looks relatively better). Magenta nodes see +m (narrower, a de-facto FPU
    reduction).
  - It also moves non-terminals relative to the ±1 terminals. It is legitimate if it matches the
    true colour advantage; see the calibration section.
- **Exploitable trick?** Inside our own tree, the opponent's nodes use our evaluator too, so we
  cannot get "FPU reduction for us and none for them" without the depth bias above.
  - The cleanest real lever is **value magnitude** (shrink toward 0). It reduces the FPU asymmetry on
    both sides and the saturation, and has the c_puct-like effect discussed above.
  - A plane-13-conditioned offset (different for al=1 vs al=2 positions) is technically possible.
    It mixes depth biases in ways I could not analyse safely in the time available, so it is not
    recommended.

### Gaps
- No published study of absolute FPU = 0 combined with [-1,1] values at ~100 sims was found. The
  conclusions rest on code reasoning plus the replica.
- I did not find current (2024–26) Lc0 FPU defaults; the wiki page is from 2019.

## Prior sharpness at c_puct 1.5 / 100 sims, and logit temperature

### Takeaway
At 100 sims the prior largely *is* the move picker. A logit lead of ≥3–4 is almost never overturned
(needs Δv > 0.4–0.5), so near one-hot priors should be reserved for forced moves. A completely flat
prior is very weak: the team measured 27%. Softmax temperature (p_scale) is roughly interchangeable
with value scale and c_puct, so tune them together.

### Cited Findings
- Lc0 PolicyTemperature default 2.2 (2019): "Higher values make priors of move candidates closer to
  each other, widening the search." CPuct default 3.0: "Higher values promote more
  exploration/wider search, lower values promote more confidence/deeper search."
  — [Lc0 options wiki](https://github.com/LeelaChessZero/lc0/wiki/Lc0-options)
- KataGo uses cPUCT = 1.1.
  — [KataGo paper](https://arxiv.org/pdf/1902.10565)
- Gumbel AlphaZero/MuZero: standard AlphaZero can fail to improve its policy when not all root
  actions are visited, and the Gumbel variants "significantly improve prior performance when planning
  with few simulations".
  — [ICLR 2022, Danihelka et al.](https://iclr.cc/virtual/2022/poster/6418); [mlanthology](https://mlanthology.org/iclr/2022/danihelka2022iclr-policy)
- A later paper notes that with a small number of simulations Gumbel cannot guarantee policy
  improvement either.
  — [Multiagent Gumbel MuZero, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29121)
- The referee mixes Dirichlet noise in after softmax normalisation, so "no scaling trick can buy a
  model its way out". It is active only for the first 8 actions.
  — [mcts.py L278-306, L588-596](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/mcts.py)
- Replica test A (override table above) and test D: the number of distinct root moves receiving the
  first 8 sims is 8 at P_top 0.3, 6 at 0.6, 5 at 0.9 and 4 at 0.97.
  — replica (root_bandit.py)
- Team sweep: p_scale 0 → 27%, p_scale 2 → 50% vs stopgap_v1 (p_scale 4 result not yet in the file).
  — [sweep_results.jsonl](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/sweep_results.jsonl)

### Inferences
- **Trade-off.**
  - Near one-hot (lead ≥ 4–6): the search only verifies the prior's choice. It catches the choice
    being a blunder only if the value of that move is clearly negative, which with FPU 0 sends
    visits elsewhere. That is safe for kernel captures and forced escapes, bad for anything else.
  - Flat: 100 sims spread over 20–30 moves (~4 visits each, plus 2-action turns), so the value's
    1-ply noise picks the move.
  - Middle ground: leads of 1–3 for "probably good" classes, so the value can overrule them with a
    0.1–0.4 advantage.
- **Temperature T on logits** divides every logit gap by T. Since U ∝ P, sharper P raises U for the
  top move roughly by e^(ΔL/T). Test C shows halving the value scale has a similar qualitative
  effect to sharpening the prior.
- **Concrete example** (root, N ≈ 100, √(N+1) ≈ 10): with a 4-logit lead, an unvisited move has
  P ≈ 0.0136 among 20, so U = 1.5·0.0136·10 = 0.20. Once the favourite's Q exceeds ~0.2, the
  low-prior move is never tried (matches test B2).
- **Dirichlet cannot be gamed**, and it only affects actions 1–8, so it does not change these
  conclusions for the decisive phase.

### Gaps
- No literature measured the optimal prior entropy at ~100 sims for hand-made priors. Lc0's 2.2
  temperature applies to trained networks, which are overconfident from visit-count training; it is
  not directly transferable.

## Tactical horizon at 100 sims with 2-action turns; baking in lookahead and search distillation

### Takeaway
The search sees roughly one full opponent reply (about 4 actions deep) on its main line, and only
1–2 actions on side lines. Most root first-actions are judged by a single al=1 leaf evaluation. The
evaluator must therefore bake in quiescence-like knowledge: 2-action capture threats, SEE-style
exchanges, kernel attackability and turn phase (plane 13). Literature on MCTS-minimax hybrids and
search distillation supports pushing shallow-search knowledge into the priors and values.

### Cited Findings
- The referee says first actions are under-searched at actions_left = 2 ("a root child must be
  visited several times before its own second action is searched at all"). The fix (adaptive
  budget) is off.
  — [mcts.py L195, L659-666](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/mcts.py)
- Baier & Winands: MCTS's selective search and averaging backups make it susceptible to traps. Of the
  MCTS-minimax hybrids they tested with evaluation functions, using (enhanced) minimax to compute
  **node priors** was the strongest, and in Breakthrough it outperformed standalone enhanced minimax.
  MCTS remains more trap-prone than alpha-beta whatever the evaluator.
  — [JAIR 2018](https://jair.org/index.php/jair/article/view/11208); [IJCAI 2018 extended abstract](https://www.ijcai.org/proceedings/2018/782)
- Implicit minimax backups (Lanctot et al. 2014): backing heuristic evaluations up as minimax values,
  separately from win rates, and blending them into selection gave stronger play in Kalah,
  Breakthrough and Lines of Action.
  — [arXiv 1406.0486](https://arxiv.org/abs/1406.0486)
- Expert Iteration: a fast apprentice policy is trained to imitate a slower tree-search expert, then
  improves the expert in turn.
  — [Anthony et al. 2017, arXiv 1705.08439](https://arxiv.org/abs/1705.08439)
- The referee's own docstring: "search visits are a strictly better policy than the raw network
  output that produced them" (the AlphaZero policy target).
  — [mcts.py L684-688](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/mcts.py)

### Inferences
- **Depth estimate** (inference, not measured): the top root edge gets ~40–60 of 100 visits. Each
  later action splits visits among ~15–25 legal moves, roughly halving or quartering per level. The
  PV therefore reaches ~4–5 actions (our second action, then the opponent's two) with a handful of
  visits.
  - "We threaten, they defend, we capture" (5–6 actions) is mostly invisible.
  - Kernel captures are the exception: they are terminal, so once visited they are seen exactly.
- **What to bake into the value** (implementable as convolutions plus ReduceMax, all allow-listed):
  - Best capture available to the side to move, using the max rather than the sum. Weight it by the
    number of actions left: one capture at al=1, about 1 + 0.5 at al=2.
  - Own pieces en prise against the opponent's coming 2 actions. These should weigh more at al=1:
    after our single action the opponent gets 2.
  - Kernel attackable with no guard as a decisive term.
  - Kernel ring vs firewall timing (already present).
  - stopgap ignores plane 13 and uses sum-of-attacked weights 1.0/−0.5. That is an approximation that
    likely overrates the mover at al=1 leaves.
- **Turn-phase consistency check** (cheap, about 30 min): on sampled positions, compare v(s) with the
  1-ply backup.
  - For al=2 positions: max over a of v(child, al=1).
  - For al=1 positions: max over a of −v(child, opponent al=2).
  - Fit a per-phase bias and correct it. Mixed-depth Q averaging means a phase bias δ at al=1 leaves
    favours lightly searched root moves by about δ/n.
- **Distilling a deeper search into the policy** is the literature-backed upgrade (ExIt, minimax node
  priors). Example: a 2–3 action alpha-beta with the hand value, giving targets for a small CNN
  policy. Within 1–2 hours a cheaper approximation is to hand-code SEE and threat terms in the prior.
  A trained distillation is only worth it if a training pipeline already exists.

### Gaps
- No direct measurement of tree depth under the real referee for this game. Logging the depth of each
  simulation in a local `mcts.run` replay would settle it.

## Value calibration: win probability vs material score; colour asymmetry and firewall phases

### Takeaway
Target 2p−1 (calibrated win probability). AlphaZero-style training converges to it, and the referee's
fixed c_puct and VL are in those units. A tanh-squashed material score is fine *if its scale is
fitted* (tanh(x) = 2σ(2x) − 1, so a logistic regression gives the scale directly). Encode the colour
asymmetry as a colour-antisymmetric term and the firewall as phase × colour terms.

### Cited Findings
- Magenta wins every tiebreak. The full lock after round 26 ends every game. If a lock destroys both
  kernels at once, Magenta wins. Plane 15 marks Magenta to move.
  — [rules.md](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/docs/rules.md); [observation.md](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/docs/observation.md)
- The search applies the firewall shrink only after Magenta's turn in a shrink round, and scores
  kernel burns at that point as terminal.
  — [mcts.py L494-510](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/mcts.py)
- KataGo blends win utility with a bounded score utility, so that large unlikely score swings don't
  dominate.
  — [KataGo paper, Appendix F](https://arxiv.org/pdf/1902.10565)
- hand_v1 already carries a "mag": 0.15 colour feature and Magenta/Cyan-specific firewall terms.
  — [hand.py](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/hand/hand.py)
- Team brief: the global Cyan win rate is ~38%. It is not independently verified here; it would put
  the calibrated starting value for Cyan at about 2·0.38 − 1 = −0.24.

### Inferences
- **Calibrated values keep Q gaps in the 0.05–0.4 band** where c_puct 1.5 and VL 1.0 behave sensibly
  (test A). Material-squashed values at stopgap scale saturate. A KataGo-style split, with a bounded
  win-probability part plus a small material tiebreak inside the same tanh, is a reasonable
  compromise.
- **Colour:** a term ±m on plane 15 is depth-consistent (see the FPU section). Fit m from data;
  0.1–0.15 matches hand.py and the ~−0.24 start prior.
- **Firewall:** since every game ends by the round-26 lock and Magenta wins the double burn, Cyan's
  value should drift negative as urgency rises, unless Cyan has a forcing kernel attack. Use a
  plane14 × plane15 term.
  - Kernel-on-next-locked-ring with high urgency should be near-decisive, because the search only
    "sees" the burn after Magenta's turn.

### Gaps
- No calibration data has been computed yet for this game. The tournament-match replays in
  `data/matches` are the obvious source.

## Virtual loss 1.0, batch 8, 100 sims: distortions and compensation

### Takeaway
VL 1.0 is a full "loss" in Q units. During each of the ~12 batches it forces descents onto different
moves, so the first 8 sims become a 1-ply scan of the top ~5–8 prior moves and the search is broader
than sequential PUCT. A model can only partly counter this: sharper top priors and larger value
magnitudes make VL relatively smaller. It can also exploit it, by making sure the right moves are in
that top-8.

### Cited Findings
- A pending descent counts as a visit with value −VL: `q = (W − VL·vl)/n`. `total` includes pending
  visits. A second descent to an unexpanded pending leaf ends the batch early, and no simulation is
  wasted.
  — [mcts.py L342-356, L624-631](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/mcts.py)
- The docstring calls sensible VL values "In Q units ... [0.5, 3.0]".
  — [mcts.py L188-190](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/mcts.py)
- Lc0 MinibatchSize: "Larger batches may reduce strength a bit, especially with a small number of
  playouts."
  — [Lc0 options wiki](https://github.com/LeelaChessZero/lc0/wiki/Lc0-options)
- Replica test D: distinct moves in the first 8 sims = 8 / 6 / 5 / 4 at P_top 0.3 / 0.6 / 0.9 / 0.97.
  — replica (root_bandit.py)

### Inferences
- **Numeric example.** A root edge with n = 1 real visit and W = 0.3, plus one pending descent, shows
  Q = (0.3 − 1)/2 = −0.35, so the next descent goes elsewhere. At n = 10, W = 3, one pending descent
  only lowers Q from 0.30 to 0.18, so a mature edge absorbs 2–3 descents per batch.
  - Distortion is largest early and on fresh nodes, which are exactly the al=1 nodes under each root
    first-action. Second actions get explored broadly rather than deeply.
- **Shrinking values makes VL relatively heavier** (VL is fixed at 1 while Q gaps shrink). That is
  part of why v_scale 0.5 acts "wider".
- **Compensation:**
  - Make the top ~8 priors the right 8. Ranking matters, as in test B3.
  - Use a sharp prior only for forced moves.
  - Note that terminal kernel captures bypass VL entirely and are backed up immediately, so a high
    kernel-capture logit lets the search pile visits on a found kill within one batch.

### Gaps
- The effect of batch 8 vs sequential search on playing strength was not measured. It could be run
  locally with `MCTSConfig(batch_size=1)` vs 8 if time allows (1 core).

## Policy logits as a move-ordering heuristic; evaluator-only contest lessons

### Takeaway
Yes. Because selection reads only P for unvisited moves and the first batch spreads by prior rank,
the logits act exactly like a move-ordering heuristic: captures by SEE, kernel kills, kernel-safety
responses, fire escapes, then positional tie-breaks. Minimax-computed priors were the strongest
MCTS-minimax hybrid in the literature. I found no published "organiser runs the MCTS,
evaluator-only" contest write-ups.

### Cited Findings
- Minimax-computed node priors gave the strongest hybrid (Othello, Breakthrough, Catch the Lion
  tests).
  — [Baier & Winands, JAIR 2018](https://jair.org/index.php/jair/article/view/11208)
- On the very first visit, `sqrt(total+1)` makes the highest prior win: "This way the highest prior
  wins the first visit."
  — [mcts.py L343-347](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/mcts.py)
- The contest's own framing: "the tournament measures *evaluation quality*, not who wrote the fastest
  tree search".
  — [submission.md](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/docs/submission.md)
- hand_v1's prior already has escape, danger-interaction, capture, capture-hanging and kernel-fire
  terms (DEFAULT_POL).
  — [hand.py](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/hand/hand.py)

### Inferences
- **Suggested logit layout** (additive from/to terms like stopgap/hand):

| Move class | Logit |
|---|---|
| Kernel capture (unguarded) | +5 |
| Capture of an undefended piece | +value/20 |
| Capture into a defended square | +(value − attacker)/20, floor 0 |
| Kernel move out of attack or out of the next-locked ring | +2.5 |
| Guard step adjacent to own attacked kernel | +1.5 |
| Moving onto an enemy-attacked undefended square | −piece value/50 |
| Quiet positional moves | 0–1 tie-breaks |

- Because the same prior drives opponent nodes, this ordering also makes the search find the
  opponent's best replies first, which is effectively threat detection.

### Gaps
- No evaluator-only competition post-mortems were found (searched indirectly). This is a genuine gap,
  not a negative result.

## Pitfalls that make a strong evaluator play badly under this PUCT

### Takeaway
The main failure modes are listed below.
- Saturated/overconfident values (lose discrimination; FPU blinds the winning side).
- Wrong sign at actions_left = 1 positions.
- Constant side-to-move offsets (odd/even bias).
- Exact prior ties (move-generation order decides what gets searched).
- Logit gaps ≥ 4 on non-forced moves.
- Values that ignore that the firewall is only scored after Magenta's turn.

### Cited Findings
- The value must be from the team to move, and within a turn the same team moves again: no sign flip
  at al=1. Plane 13 = actions_left/2.
  — [mcts.py L431-438](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/mcts.py); [observation.md](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/docs/observation.md)
- A node with no legal actions caches the model's value as `leaf_value` and reuses it on every visit.
  — [mcts.py L290-292, L402-404](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/mcts.py)
- The root's own evaluation value is discarded (`_expand_root` returns it and `run` ignores it).
  Only the root priors matter.
  — [mcts.py L597](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/mcts.py)
- Team sweep: v_scale 2 → 32% overall and **10% as Cyan**.
  — [sweep_results.jsonl](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/sweep_results.jsonl)

### Inferences
- **Sign check for any trained value:** corr(v(parent al=2), v(child al=1)) must be strongly
  positive, and corr(v(parent al=1), v(child opp al=2)) strongly negative. An "alternating-turn"
  training assumption flips the al=1 sign and is catastrophic.
- **A constant value** (e.g. v ≡ 0) turns the search into prior-proportional visits plus terminal
  detection only. A **constant prior** turns it into a noisy 1-ply value player (the team's p0
  result, 27%).
- **The v_scale 2 collapse being Cyan-specific** fits saturation plus FPU. Cyan is usually behind
  (start ≈ −0.24), its values sit at large negative magnitudes, so its own nodes go maximally wide
  and shallow, while Magenta's nodes in the same tree narrow.
- **Validate every variant** with `tools/validate_submission.py` and compare on ≥200 colour-balanced
  games. Gaps under ~10 pp are noise at 60 games.

### Gaps
- No direct data on how often stopgap/hand values exceed |0.9| in real games. A histogram over
  replayed positions would quantify the saturation risk.

## Ranking of levers by expected strength gain (summary, inference)
1. **Prior ordering of critical moves**: SEE captures, kernel safety responses, fire escapes, and
   tie-breaks so the right moves sit in the top ~5. The team's p0 = 27% shows the prior is the
   biggest lever, and tests B2/B3 show rank inside the top-8 decides discovery.
2. **Quiescence-like, plane-13-aware value**: max-capture-available, 2-action en-prise, kernel
   attackability, per-phase bias correction. This addresses the ~4-action horizon and the
   under-searched first actions.
3. **De-saturate the non-decisive value**: soft×0.4–0.6, decisive terms kept at ±3 pre-activation,
   ideally fitted by logistic calibration. Avoids FPU blinding and tanh compression; v2 → 32%
   confirms the direction.
4. **Colour-antisymmetric bias plus Cyan time-pressure term** (plane 15, plane 14 × 15): small,
   clean and depth-consistent.
5. **Avoid**: side-to-move constant offsets, logit leads ≥ 4 on non-forced moves, exact prior ties.
