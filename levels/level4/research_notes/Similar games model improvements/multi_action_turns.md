# Multi-action turns: how engines evaluate mid-turn positions and detect turn-spanning threats, and what transfers to COLONY: ONE

Scope: Arimaa (4 steps/turn), Marseillais / Duck chess (2 legs/turn), and what fits a PyTorch->ONNX evaluator under the referee's fixed per-action PUCT (100 sims, FPU 0, sign flip only at turn change). Written 2026-10-23 ~14:50 UTC.

Primary source: David Wu, "Designing a Winning Arimaa Program", ICGA Journal, March 2015 (the bot_Sharp paper). The full text was read from the PDF: [Wu 2015 PDF](https://icosahedral.net/downloads/djwu2015arimaa.pdf), [ICGA abstract](https://ip.ios.semcs.net/articles/icga-journal/icg38104).

Local context:
- `levels/level4/agents/hand/fast4.py` (hand_v4 graph). It already has the 2-action reach maps `A2`/`hit2`, the value features `own_hang2_*` / `en_hang2a2_*` / `*_g1_t2`, and the defensive prior terms `danger2` and `k_d2`.
- `levels/level4/agents/*/RESULTS.md` and `levels/level4/reviews/hand_v2.md`.

**Recommendations below are for things hand_v4 does NOT already have.**

---

## Q1. Arimaa top bots (Sharp, Bomb, etc.): threat detection over the remaining steps of a turn, steps-left-aware evaluation, and neural attempts

### Takeaway
Strong Arimaa bots do not search out 4-step threats. They detect "goal in up to 4 steps" and "piece capturable within the steps left" with direct pattern code, and use it at leaves, in move ordering and in quiescence. Sharp's biggest search gain was a set of forward "tactical move generators" that propose the full multi-step tactic (quiet steps plus capture) as one unit: +80 Elo. Its evaluation scores piece threats by the number of steps the capture needs. It also combines a linear eval with explicit short-term win and loss probabilities instead of letting threats add up linearly.

### Cited Findings
**Static detection instead of search**
- Fotland and Zhong were among the first to do well with alpha-beta and pioneered ideas "used by nearly all strong Arimaa programs today, such as static capture and goal detection". — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf); also [Wikipedia: Computer Arimaa](https://en.wikipedia.org/wiki/Computer_Arimaa)
- Static goal detection:
  - Checking whether the side to move can score a goal this turn "naively ... would appear to require a four-step search". Instead, Fotland's approach enumerates the classes of patterns that allow a goal in 4 steps and tests them directly with specialized code (e.g. a decision tree), "a huge speedup over a search".
  - Sharp has "several thousand lines of code" for goal-in-up-to-4-steps. It is cheap enough to call at leaf nodes, "extending the effective search depth by a full four steps for goal threats". — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)
- Static capture generation: "all strong Arimaa programs contain code for statically detecting whether a piece is capturable and for generating capturing moves." In Sharp it is used in quiescence search and to improve move ordering. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)

**Quiescence and extensions depend on the steps left**
- Sharp's quiescence is capped at 3 moves (12 steps). The move classes per layer depend on the steps left in the current turn:
  - Layer 0 with 1-3 steps left: goal defense, captures, goal threats, capture defense, **capture threats**, pushes/pulls of trap defenders, other tactical moves.
  - Layer 0 with 4 steps left: goal defense, captures, goal threats, capture defense.
  - Layers 1-2: goal defense, captures, rare moves only. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)
- Extensions: +1 step when a goal threat is played whose minimum defense needs >= 2 steps, +2 steps when it needs >= 3 steps. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)
- Goal relevance pruning: with n steps left, steps that only touch squares more than 3(n-1) away from the squares involved in a goal threat can be excluded from the defenses. A tighter radius (2(n-1), or n when freezing is irrelevant) applies in some cases. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)

**Tactical move generators (largest single gain)**
- They are "the most significant single innovation in Sharp's search in 2015" and gave **+80 Elo**, "the largest gain any single change to the search has produced in the last several years".
- They produce whole multi-step tactics: up-to-4-step pushes/pulls next to traps, runaway or unfreeze moves for threatened pieces, adding trap defenders, "likely forced wins-in-2", and so on.
- They catch about 97% of the evaluation difference between expert moves and passing, while generating only about 3% of legal moves. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)
- How they were tuned: Wu did **not** maximize the fraction of expert moves generated. He "minimized the amount by which Sharp's evaluation of the best move produced by any generator was worse than Sharp's evaluation of the expert move". The difference was treated as 0 when the generator's move was better, capped, and measured with a quick 5-step search. Quiet positions where "the precise choice of move does not make a major tactical difference" were deliberately not targeted. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)

**Evaluation**
- Piece-threat terms are products, `Value(p) * f(TrapScore(p)) * g(ThreatDistance(p)) * ...`. Features include "the number of steps required to capture the piece, including steps to push away defenders".
- The piece's value is computed as the material score with and without it.
- "To avoid overpenalizing" a piece that is already penalized by another feature, **only the maximum of the two penalties is used**. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)
- For each piece, Sharp computes the steps (0-5) needed to free it and the steps (0-5) the opponent needs to dominate it. These feed most other evaluation components. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)
- Volatility model: the goal-threat component outputs short-term win probabilities g (Gold) and s (Silver). The linear eval x becomes p = sigmoid(kx), and **Final = g + p*(1 - g - s)**.
  - Rationale: linear evals "happily take the offered material" during a goal attack.
  - Measured gain: **about +15 Elo** in self-play. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)

**Move ordering**
- Root: a generalized Bradley-Terry model trained on expert games. Features include source/destination and piece type, captures, capture/goal threats and defense, trap defense, and "step dependency structure".
- It finds the expert move in the top 1% of moves more than 80% of the time, and in the top 20% more than 99% of the time. It is too slow (0.1 s) to use below the root. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf); secondary summary of the 2011 thesis: [Wu thesis](https://econcs.seas.harvard.edu/resource/dwuthesispdf)
- In-tree order:
  1. TT move.
  2. 8 killers per ply, stored as full 4-step moves.
  3. Captures, ordered by the size of the captured piece.
  4. Tactical moves, ordered by history + move type.
  5. TT-move prefix (reduced by 1 step).
  6. Other steps, ordered by history (reduced by 1 step each).
- The history table is indexed by the **first step** of the move (56*4), or the first two steps for pushes and pulls. There is a separate table per ply. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)

**Results and other approaches**
- Sharp gained more than 600 Elo in blitz over two years and won the 2015 Arimaa Challenge. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf), [ICGA abstract](https://ip.ios.semcs.net/articles/icga-journal/icg38104)
- Sharp's evaluation is hand-written: "thousands of lines", hundreds of parameters, tuned by self-play over hundreds of games with BayesElo. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)
- A 2012 IPSJ report tested TD, TD-Leaf, Treestrap and Rootstrap for Arimaa eval weights. Search-snippet only; not read. — [IPSJ GI12027004](https://ipsj.ixsq.nii.ac.jp/record/80932/files/IPSJ-GI12027004.pdf)
- MCTS in Arimaa (Kozelek 2009 thesis): per a secondary page, the results were "disappointing" and only on par with mid-level alpha-beta bots. The author attributed this to the alpha-beta bots' strong evaluations. Full text not accessed; the fetched page was empty, so this comes from the search snippet only. — [Kozelek thesis record](https://dspace.cuni.cz/handle/20.500.11956/30658), [TU Berlin wiki](https://wiki.neuro.tu-berlin.de/home/teaching/2024ws-AlgoDat-SE/WikiGames/Arimaa)
- Neural bots: Wikibooks says neural-network Arimaa bots appeared after AlphaZero and names "Rusty_Zero" as the top bot (about 3000 Elo). — [Wikibooks Arimaa Overview](https://en.wikibooks.org/wiki/Arimaa/Overview)

### Inferences
- What transfers directly:
  - **Static per-piece capture detection within the actions left.** This is hand_v4's `hit`/`hit2`.
  - **Steps-left-dependent treatment.** What counts as a "threat" differs for 1 action left vs 2.
  - **Max, not sum,** over overlapping threat penalties.
  - **Generating or ordering whole multi-action tactics,** not just their first step.
- Sharp's tuning criterion ("minimize eval regret vs the reference move, ignore quiet positions") is a training-data change we can copy for prior and value distillation. See Q4, item P1.
- The g/s mixture is the most principled way to stop linear threat sums from saturating. But Sharp measured only +15 Elo, and the hand team already found hard win_now/win2 reweighting neutral. Low priority.

### Gaps
- I could not find how Rusty_Zero or other neural Arimaa bots were designed (per-step vs per-move policy head, whether they used static-capture input features). No primary source found.
- I could not read Kozelek's thesis, so the tree structure (step or move nodes) of the Arimaa MCTS attempts is unconfirmed.

---

## Q2. Marseillais (double-move), Progressive, Monster and Duck chess: engine techniques

### Takeaway
The available sources show the same design throughout. Engines represent each leg of a multi-leg turn as its own ply and flip the side to move only after the last leg: ChessV for Marseillais, H.G. Muller's design for Duck chess. Whole-turn search is avoided because of the branching factor (about 40x40 move pairs). Duck chess shows how to handle a leg that cannot capture: score the turn by minimax inside the node, and use the second leg mainly to block the opponent's best reply. I found no source on two-move-mate detection or specialized threat evaluation for these variants.

### Cited Findings
- Marseillais chess: each player moves twice per turn (first published in 1925). Checking with the first leg forfeits the second, and a player in check must escape with the first leg. — [Wikipedia: Marseillais chess](https://en.wikipedia.org/wiki/Marseillais_chess)
- "A correct search in a 'Marseillais Chess' engine needs to find the best 'move pair'", which can mean up to about 40x40 pairs in a 40-move position. — [Talkchess thread](https://talkchess.com/viewtopic.php?p=556239)
- ChessV handles multi-move variants by treating each leg as a separate move and adjusting when the side to move flips. As of 2019, the commenter knew of no other xboard engine that played Marseillais. — [chessvariants.org comment, G. Strong](https://www.chessvariants.org/index/listcomments.php?id=38026)
- Duck chess (H.G. Muller):
  - "the search tree consists of levels that alternately move a FIDE piece and a Duck". Within a combined node "we must minimax the scores within the node".
  - The Duck move "cannot be a capture" and its only function "is to prevent the opponent's best move in the next ply".
  - After the first ply, all Duck moves are transpositions. — [Talkchess duck chess thread](https://talkchess.com/viewtopic.php?p=936168)
- Fairy-Stockfish lists Duck chess among its supported variants. — [Fairy-Stockfish GitHub](https://github.com/ianfab/Fairy-Stockfish)

### Inferences
- Our referee's tree (one node per action, sign flip only at turn change) is the standard design used by these engines. So nothing in the tree is "wrong". The burden falls on the value at the intermediate node and on the prior at both legs.
- Duck-chess lesson for us: at a 1-action-left node, the decisive question is often "which single action prevents the opponent's best 2-action reply". That is a defensive-relevance prior; hand_v4 already has `danger2` and `k_d2`.

### Gaps
- No source found on Fairy-Stockfish support for Marseillais or Progressive chess, or on any special double-move evaluation in it. `variants.ini` was not checked because of time.
- Progressive chess (n moves on turn n) and Monster chess: no engine-technique sources retrieved within the time budget.
- No source found on "two-move mate" static detection in Marseillais engines.

---

## Q3. Scoring a mid-turn position (1 action left) vs a turn-start position (2 left); 2-action reach maps; computing them with fixed convs in ONNX

### Takeaway
A mid-turn position needs a different static threat accounting than a turn-start position. With 1 action left:
- the side to move can execute only 1-action captures;
- the opponent then gets a full 2-action turn;
- so our pieces under enemy 2-action reach are really threatened, while our own 2-action threats are worth little.

With 2 actions left it is the reverse. Each threat feature therefore needs separate weights for 1 and 2 actions left (gated features), and threats should be aggregated by max or top-2 (one strike per turn), not summed. hand_v4 already computes the hard part, the 2-action reach map via a second ray pass. What is missing is (a) gating `own_hang2` by actions left, (b) max/top-2 aggregation and a "safe capture" (static-exchange-lite) refinement, and (c) per-piece-type strike maps for the prior (Q4).

### Cited Findings
- Sharp's quiescence includes capture threats only when 1-3 steps are left, and uses a different set at 4 steps left. Its search extensions depend on the defense's minimum step count. So threat handling is explicitly conditioned on the steps left. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)
- Sharp's piece-threat eval uses "the number of steps required to capture the piece" and takes "only the maximum of the two penalties" when features overlap. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)
- Sharp stresses that static capture/goal detection replaces a multi-step search by pattern tests cheap enough to run at every leaf. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)
- Local, hand_v4 (`agents/hand/fast4.py` lines 247-254):
  - `A2` is the attack map after one quiet move. It is built from `MV` (quiet-move destination maps per type), a second shift-conv ray pass with `*empty` blocking (3 steps), and a 5x5 conv for the Exploit and Kernel.
  - `hit2 = clamp(A2) * (1 - hit)`.
  - `H2` contains `own5 * hit2_en * (1 - hit_own)` (feature `own_hang2_*`, gated `one` only) and `en5 * hit2_own * (1 - hit_en)` (feature `en_hang2a2_*`, gated `a2`).
  - All of these are **sums over squares**.
- Local results:
  - hand_v4f (the v4 bundle plus value refit): 109/200 = 55% vs hand_v3.
  - hand_v4p (v3 value plus v4 prior plus hand-set g1_t2): 76/200 = 38%.
  - The 2-action maps have so far given only a small gain. — `levels/level4/agents/hand/RESULTS.md`

### Inferences
**Why "sum over hanging pieces" is wrong with 2 actions per turn**
- The striker can usually cash only one 2-action strike, or two 1-action captures, per turn. The defender with 2 actions can usually fix one or two threats.
- So the material at risk is roughly:
  - **1 action left, threats against the side to move:** after our 1 action the opponent has 2 actions. Expected loss is about max(top1(hang1) + top2(hang1), top1(hang2)) minus what our single action saves (about top1).
  - **2 actions left, threats against the side to move:** two defensive actions; the loss is mostly forks (3rd-largest 1-action threat, or a 2-action threat plus another).
  - **Own capture chances, 2 actions left:** top1(en_hang1) + top2(en_hang1), or top1(en_hang2).
  - **Own capture chances, 1 action left:** only top1(en_hang1).
- A linear sum cannot express this. Texel-fitting `max` and `2nd` features gated by a1/a2 lets the fit discover it.

**Game-specific static-exchange refinement**
- Within our turn the opponent does not move between our two actions. Only the square where our striker ends must be safe at turn end:
  - **Move-capturers** (Ping, Ice, Daemon at distance 2-3, Exploit, Kernel) end on the victim square, so a strike is "safe" if `1 - hit_en(victim_sq)` after the capture. Approximation: `(1 - hit_en)` at the victim square, excluding the victim itself as a defender.
  - **Shooters** (Daemon adjacent shot, Trace diagonal 2-3) and **guard kills** (an attack on a guarded Kernel) leave the attacker on its square. So the safety of the setup square matters, not the victim's.
- hand_v4's `(1 - hit_own)` and `(1 - hit_en)` factors use "victim undefended". That treats a defended victim as never winnable, even when the captor is worth less or is a shooter that cannot be recaptured. A cheap fix is to add shooter-only maps: `en5 * hitShoot_own`, defended or not.

**What hand_v4 lacks for the a1 vs a2 split**
- The gate algebra already spans a1 vs a2 when a feature exists with both the `one` and `a2` gates. `own_hang2_*` exists only with `one`, so the fit cannot separate "opponent strikes after my 1 action" from "after my 2 actions". Adding the `(own_hang2_t, "a2")` FMAP entries is a 5-minute change plus a refit.
- The same applies to the 1-action `own_hang_*` and `en_hang_*`. Check that both gates exist.

**ONNX cost**
- Everything above is ReduceMax over (H, W) of products of existing maps.
- Top-2 can be done with `TopK` (opset 17), or approximated with sum and max.
- That is about 20-40 extra nodes, well within the caps (hand_v4: 238 nodes and 286 initializers, against 1024 and 512).

#### Ranked implementation sketches (expected gain per hour of team time, highest first)

**1. [Data, about 30-45 min, distill or prior pipeline] Regret-weighted targets plus "after-quiet-move" child samples.**

Evidence: distill's own note says "more data and the child value samples are what help". Sharp tuned generators by eval regret, not hit rate.

Steps:
- (a) For each 2-actions-left training position, add child samples at 1 action left: the position after each of the top-k teacher first actions, especially quiet ones. The value target is the teacher score of the best second action, with captures-only quiescence. This directly teaches the a1-node value "I can now capture X". In the referee MCTS that value is what makes the first visit of a quiet setup move return a high Q, so PUCT keeps visiting it despite a low prior.
- (b) Weight the policy cross-entropy per position by `clip(Q_best - Q_second, 0.05, 0.5)` or by `Q_best - Q(100-sim choice)`. Quiet positions then stop dominating the gradient (median top-1 prior is 0.115, and 46% of positions have top-1 < 0.10, per hand_v2.md W5).

Applies to distill_v3hb (78% vs hand_v3, the current best) and to the prior agent.

**2. [Value features, about 20-30 min incl. refit, hand_v4 / FastNet4] a1/a2 gates plus max/top-2 threat aggregation plus shooter-safe captures.**

```python
# per side, per type maps already exist: own5, en5 (piece one-hots), val vector
# Wv = [vP, vD, vX, vT, vI]
ownVal = (own5 * Wv).sum(1, keepdim=True)        # (B,1,8,8)
enVal  = (en5  * Wv).sum(1, keepdim=True)
h1o = ownVal * (hit_en - D)                       # my undefended pieces the enemy hits now
h2o = ownVal * hit2[:, 1:2] * (1 - hit_own)       # ... hits after one quiet move
h1e = enVal * (hit_own - D); h2e = enVal * hit2[:, 0:1] * (1 - hit_en)
top2 = lambda m: torch.topk(m.flatten(1), 2, dim=1).values   # (B,2)
f = cat([top2(h1o), h2o.flatten(1).amax(1, keepdim=True),
         top2(h1e), h2e.flatten(1).amax(1, keepdim=True)], 1)  # 6 scalars
# gate each by a1 and a2 -> 12 features; Texel-refit with fit4.py
```

Optionally add `shoot_own` (Trace and Daemon shot maps only) times `enVal`, ungated by defence.

Expect a small gain (v4f went to 55% with sums). The value is in fixing the gating bug for `own_hang2`. Low risk: a refit with the existing harness, then a 200-game arena (about 5 min CPU at about 1.3 s per game).

**3. [Prior, about 30-45 min, FastNet4 policy] Offensive "strike-setup" term.** See Q4, P2. It is the only part of W2 with no prior signal in v4.

**4. [Residual students, about 15 min plus retrain] Feed `hit2`/`A2` and the per-type strike maps as extra input planes** to the az_res and distill trunks (32ch x 2-4 res blocks).

- A 2-action strike spans up to 3 (move) + 3 (capture) squares.
- A small trunk can represent it in principle (5 convs of 3x3 give an 11x11 receptive field), but learning ray-blocking from a few thousand positions is slow.
- Explicit maps are the "static capture detection" that every strong Arimaa bot hand-codes.
- Only worth it if a student is being retrained anyway.

**5. [Value, about 15 min] Sharp's mixture output**, `v = w - l + (1 - w - l) * tanh(x)`.
- `w = clamp(win_now + a2 * g0_en * t2c_en)`: a forced Kernel capture within the actions left (as in v4's `win2`).
- `l` = the opponent's unstoppable equivalent at a1 (`g0_own * (ka_own + t2c_own) * a1`, roughly).
- Sharp measured +15 Elo, and hand team reweighting of these terms was neutral. Do it last, if at all.

### Gaps
- No source measured how much a1-specific value features help under a per-action MCTS. The ranking above is inference from Sharp's design and from the team's own results.
- The exact referee FPU semantics (absolute 0 vs parent-relative) were not re-checked. The claim in item 1 that "the first visit's value drives exploration" assumes FPU is an absolute Q of 0 for unvisited children.

---

## Q4. Policy / move ordering for multi-action turns, and its translation into from-square x to-square prior logits

### Takeaway
Sharp's in-tree ordering has four parts: captures ordered by victim value; "tactical" generators that emit whole multi-step tactics; full-length killer and TT moves; and a history table keyed on the first step. Its root ordering is a feature-based (Bradley-Terry) predictor that includes threat and dependency features.

In our per-action tree, the analog is a two-part prior:
- at 2 actions left, put mass on the **first leg of a strong 2-action plan** (a quiet move that enables a capture, or a defense against an enemy 2-action threat);
- at 1 action left, put a sharp peak on the completion (the capture).

hand_v4's prior already rewards immediate captures (`to = enV2 * cap`) and defense (`danger2`, `k_d2`, `guard`, `k_esc_thr`). It has **no term for "this quiet move sets up my capture next action"**, because its rank-5 bilinear form is piece-type-agnostic.

### Cited Findings
- In-tree ordering: TT move, then killers (8 per ply, "always recorded with the full four steps"), then captures "ordered by size of piece captured", then tactical moves, then other steps by history + move type with depth reductions. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)
- History heuristic keyed by the move's first step, or first two steps for pushes and pulls, with "a separate table ... for every ply". — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)
- Static capture generators "generate the entire capture sequence rather than merely the step that begins the capturing move". — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)
- Root Bradley-Terry ordering uses features such as "capture and goal threats and defense, trap defense, step dependency structure". — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)
- Generators were tuned to minimize eval regret against the reference move, not to maximize coverage of quiet moves. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)
- Local results:
  - The prior is flat in quiet positions (median top-1 0.115; the 800-sim choice is the prior's top-1 in only 40% of cases). — `reviews/hand_v2.md` W5
  - Defensive 2-action prior terms plus tie-breaks were neutral (hand v4a: 104/200 = 52% vs v3). — `agents/hand/RESULTS.md`

### Inferences
**P1. Regret-weighted prior distillation.** Q3 item 1(b): Sharp's tuning objective applied to the prior agent's cross-entropy.

**P2. Offensive strike-setup prior (about 30-45 min).** A rank-6 bilinear term added to `logits[f, t]`:

```python
# For each own piece type X (P,D,X,T,I,K): S_X(t) = best victim X could hit from t next action
#   S_X = sum_k conv(enValSafe, Krev_{X,k}) * conv(empty, Kmid_{X,k})   (mid-square emptiness for range-2/3)
#   enValSafe = enVal_withK * (1 - hit_en_after)   # move-capturers: victim square safe-ish at turn end
#   for Trace/Daemon shots use enVal_withK without the safety factor (shooter stays put)
#   enVal_withK: guarded Kernel -> value of priority guard (v4's ek_guard idea), unguarded -> ek_val
L_X(f) = own_X(f)                                        # one-hot of the moving piece type
R_X(t) = a2 * empty(t) * S_X(t) * (1 - danger1(t))       # quiet move to t, t not hit by enemy now
logits += beta * sum_X L_X(f) * R_X(t)                   # matmul [B,64,6] x [B,6,64]
```

- `Krev_{X,k}` are the reversed capture offsets: Ping diagonal 1-2, Daemon adjacent shot plus orthogonal 2-3 move-capture, Exploit knight, Trace diagonal 2-3, Ice 8 directions 1-2, Kernel adjacent.
- `Kmid` is a single-1 kernel at the intermediate square(s), for blocking. Use one conv per (type, offset group). The depthwise shift machinery in `fast4.py` already does this for `A2`, with the direction reversed.
- `danger1(t)` matters only weakly, because the opponent does not move before our second action. Keep the factor small, or apply it only for shooters.
- Start with beta around 0.4-0.8 (`cap` is 1.34), or fit beta with `fit4.py --pol` on 800-sim visits.
- Gate by `a2` so it never fires at 1 action left, where the capture term itself takes over.
- Expected effect: in the 6509 a3-a4 and a7-a8 type strikes (TR H6->H7 then H7xE4; EX E7->D5 then D5xC7), the setup moves get prior mass instead of about 0.02.
- Risk: more prior mass on aggressive quiet moves into nothing. The safety factors and the max-victim (not sum) form limit this.

**P3. Completion peak at 1 action left.** After a setup move, the capture should dominate the a1 prior so that most of the 100 sims under that child back up the post-capture value. The existing `cap * enV` already gives about 5.9 logits for Kernel attacks and less for pieces. Check the median top-1 at a1 nodes that follow a setup move. If it is below about 0.5, scale the `to` capture term by `(1 + c * a1)`.

**P4. "History per ply" analog.** Sharp keeps separate statistics for each ply within a turn. Our equivalent is to let every prior term carry an a1/a2 gate. FastNet4's prior terms are mostly ungated except `warm`, `u1` and `u2`. Cheapest version: two scalar multipliers per prior row (`Rp` rows times `(alpha_k + beta_k * a2)`), fitted on visits.

### Gaps
- No published neural-net work found on policy heads for multi-step turns (per step vs per turn, or how to factor from-to priors across legs).
- No direct evidence that the "enables a capture" ordering helps under MCTS-100. Sharp's evidence is for alpha-beta generators (+80 Elo).

---

## Q5. Evidence on tree nodes per action vs per full turn, and turn-level transposition

### Takeaway
Sharp went from a pure stepwise tree to a hybrid one: nodes still mostly hold single steps, but move lists mix 1- to 4-step partial moves, so full tactics are tried without committing to a prefix. That gave about a 20% speedup at 9-step depth and enabled late-move reduction, its key selectivity gain. Pure whole-turn (movewise) search was rejected because of a branching factor around 10,000.

The referee fixes our tree at one node per action with no transposition merging. The transferable lesson is therefore about the evaluator: make the first-leg prior and the mid-turn value encode the full-turn plan, so the per-action tree behaves like a hybrid one.

### Cited Findings
- Movewise formulation: branching factor about 10,000. Stepwise formulation: about 30. Earlier work often favored stepwise because of the overhead of large move lists. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)
- Stepwise flaw: depth-first search "forc[es] the search to consider moves that begin with the same steps together". If a promising 4-step move fails, the search must "uselessly explore every other move in the subtree beginning with" its first step. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)
- Hybrid fix (2014): move lists mix 1- to 4-step moves; TT and killer moves are recorded as full 4 steps; capture generators emit whole sequences. Measured "about a 20% speedup for 9-step searches and much more for deeper searches", and it enabled effective late-move reduction. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)
- Transpositions "are common" in Arimaa because step permutations reach the same state. Sharp uses a large TT with 128-bit entries. — [Wu 2015](https://icosahedral.net/downloads/djwu2015arimaa.pdf)
- Duck chess: after the first ply all Duck moves are transpositions; one design merges nodes by FIDE placement. — [Talkchess duck chess thread](https://talkchess.com/viewtopic.php?p=936168)
- ChessV: legs as separate moves, with the side-to-move flip delayed. — [chessvariants.org comment](https://www.chessvariants.org/index/listcomments.php?id=38026)
- Local: the distill teacher (alpha-beta over single actions plus TT plus captures-only quiescence) beats stopgap_v1 19/20 in direct play, so per-action search with a TT and quiescence is strong in our game. — `agents/distill/RESULTS.md`

### Inferences
- In the referee's per-action MCTS, the stepwise flaw shows up as visit dilution. A setup quiet move with a low prior gets few visits, and each visit's value is the static a1 value, which (pre-fix) does not see the follow-up capture. The strike stays invisible: W2.
- Two evaluator-side fixes mimic Sharp's hybrid search:
  1. an a1 value that already includes "best capture available now" (Q3 item 1, Q3 item 2);
  2. a first-leg prior keyed to 2-action plans (Q4 P2).
- Turn-level transposition (A then B equal to B then A for two independent quiet moves) cannot be merged by us. It wastes some of the 100 sims in quiet positions. No evaluator fix is worth the time. A peaked prior reduces the waste indirectly.
- Distillation targets should come from a turn-aware teacher. The distill teacher already searches the rest of the own turn plus the full opponent turn plus quiescence, which is exactly the Sharp-style "full-turn tactic" signal. More data from it, particularly a1 child samples, is the cheapest way to move the per-action prior toward turn-level plans.
- Out of scope in the remaining time: changing the tree, adding a TT, or per-turn macro-actions (all fixed by the referee). Also out of scope: a learned Bradley-Terry root model separate from the net, and a full static exchange evaluator with recapture chains.

### Gaps
- No source compares per-action vs per-turn nodes for **MCTS/PUCT** specifically. The only quantitative evidence is Sharp's alpha-beta (+20% speed, LMR enabled).
- No AlphaZero-style result for a multi-action-per-turn game was found within the time budget.
