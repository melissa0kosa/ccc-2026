# Other small tactical games and bot contests: techniques NEW relative to our earlier reports

Context (15:20 UTC):
- Live model: az_res4f_v2 (hand_v4f base + residual).
- Its training data already includes 4000 endgame-start self-play games with outcome-only values (26% of samples), per `agents/az/RESULTS.md`.
- The remaining weakness is low-material (K+<=1 vs K+<=1) value. It is worst as MAGENTA (`reviews/az_res_v6.md` §3-4b).
- Already covered elsewhere and not repeated here: Arimaa, Marseillais chess, Stockfish king safety/SEE/Texel, Pommerman, Battlesnake, NNUE data generation, CrazyAra, generic AZ/KataGo/Gumbel tricks.
- Note: the earlier `arena_ranged_colony.md` said "no tablebase beyond the closed-form bare rule is needed". That holds for the *terminal* rule only. The K+1 vs K+1 *pre-terminal* play (lone guard shot, ring race) is what still loses games, and Q4/Q5 below address it.

## Q1. Race / "closer to the centre wins" endgames: how King-of-the-Hill and Racing-Kings engines score the race

### Takeaway
Variant Stockfish engines score the race to the goal region in two ways:
- as a **very steep, roughly exponential table lookup** on distance, not a linear term;
- with **enemy attacks on the path or target counted as extra distance**.

Our planned "lowmat × (ring_depth own − enemy)" linear feature has the wrong shape. Use bucketed (one-hot) depth differences, and add attack-on-path increments.

### Cited Findings
- **Multi-variant Stockfish KOTH.**
  - Table: `KothDistanceBonus[6] = { S(1949,1934), S(454,364), S(151,158), S(75,85), S(42,49), S(0,0) }`, plus a space term `KothSafeCenter = S(163,207)` per safe centre square. — [ddugovic/Stockfish evaluate.cpp](https://github.com/ddugovic/Stockfish/blob/master/src/evaluate.cpp)
  - Each step of distance costs about 4×, then about 3×, then 2×. The midgame and endgame values are almost equal, so the bonus is applied in every phase, not only in the endgame. — [ddugovic/Stockfish evaluate.cpp](https://github.com/ddugovic/Stockfish/blob/master/src/evaluate.cpp)
- **KOTH distance per centre square.** `dist = distance(king, s) + popcount(attackers_to(s) & enemy) + !!(own piece on s) + (own pawn on s blocked by enemy pawn)`, then `score += KothDistanceBonus[min(dist-1, 5)]`. Enemy attackers and own blockers on the target count as extra steps. — [ddugovic/Stockfish evaluate.cpp](https://github.com/ddugovic/Stockfish/blob/master/src/evaluate.cpp)
- **Racing Kings.**
  - Table: `KingRaceBonus[RANK_NB] = { S(14282,14493), S(6369,5378), S(4224,3557), S(2633,2219), S(1614,1456), S(975,885), S(528,502), S(0,0) }`.
  - The index starts at the king's distance to rank 8. It gets +1 for every rank ahead in which no square of the king's reachable span is free of enemy attacks, i.e. a blocked rank counts as an extra step.
  - King-danger parameters are all zero in this variant.
  - [ddugovic/Stockfish evaluate.cpp](https://github.com/ddugovic/Stockfish/blob/master/src/evaluate.cpp)
- **Fairy-Stockfish capture-the-flag (its generalised KOTH).**
  - It traverses all paths of the flag piece to the flag region, and puts squares that are **attacked or occupied "on hold for one iteration"**: "a move will be needed to block or capture the attack".
  - With all pieces eligible it uses the **king path as the distance proxy**.
  - The score is `make_score(4000,4000) * wins / (wins + dist*dist)`, where `wins` is the number of flag pieces that can reach the region.
  - There is also a space bonus `make_score(200,200) * popcount(behind & safe & flag_region)`.
  - [Fairy-Stockfish evaluate.cpp](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/master/src/evaluate.cpp)
- **Fairy-Stockfish `flagMove` option.** "the other side gets one more move after one reaches the flag zone". This tempo rule is configurable. Engines treat the race's who-moves-last parity as part of the rules, not of the eval. — [Fairy-Stockfish variants.ini](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/master/src/variants.ini)

### Inferences
- **The shape lesson.**
  - In all three engines the race term falls off steeply: geometrically in KOTH and Racing Kings, as 1/(1+d²) in Fairy-SF.
  - A single linear weight on (own ring − enemy ring) cannot both be decisive at a difference of ±1 and small at large differences. **Use one-hot buckets** of the depth difference, {≤−2, −1, 0, +1, ≥+2}, and let the refit set the five weights.
  - Bucket 0 takes the Magenta tie sign (+ if we are MAGENTA, − if CYAN). That colour asymmetry is exactly what the review's calibration table shows we under-model: MAG-lost mean v of −0.26 vs CYAN −0.86.
- **The path lesson.**
  - Our losses have the Kernel *retreating to the G6/G7 corner*, an outer ring, on rounds 8-9.
  - A KOTH-style term would count that as a large, steep loss, and would add +1 for each enemy shot covering the Kernel's inward path.
  - It can be built in ONNX without BFS code:
    1. Start from the target-ring mask.
    2. Repeat k = 1..4 times: `MaxPool 3x3, stride 1, pad 1` (the Kernel's one-step reach), then `× (1 − enemy_attack_plane)`.
    3. The Fairy-SF "on hold" trick: an attacked square is admitted one iteration late (`next = dilate(cur) * safe + dilate(dilate(cur) * attacked) * …`). A simpler version uses a "safe-only" dilation and an "any" dilation and takes the gap.
    4. The distance is the first k at which the Kernel's square is reached.
  - That is about 8-12 nodes, well under the 1024-node cap (az_res4f is 278).
- **Phase.** KOTH applies the term in all phases (mg ≈ eg). Our hard `lowmat` gate is safer for the gate test, because it gives exact parity outside the endgame. Tapering by material instead is a post-contest experiment.

### Gaps
- I did not find a published Elo measurement of these KOTH constants. They look hand-tuned or SPSA-tuned; no source states which.
- Fairy-SF's ring geometry differs from ours. Our rings shrink at fixed rounds 8/16/24/26, which is closer to a moving flag region. No engine source handles a *shrinking* goal region.

## Q2. Bare-king / extinction rules (Shatranj baring, extinction variants): how engines evaluate "last piece standing"

### Takeaway
Fairy-Stockfish treats "lose when a piece class goes extinct" with two devices:
- a threat bonus that grows as **1/(pieces above the extinction count)²**;
- an option to treat the **last remaining piece of that class as pseudo-royal**, scored like a king.

With our bodyguard rule, a lone guard is effectively pseudo-royal. The king-danger form (attacks over evasions) should apply to it, not a flat "hanging" term.

### Cited Findings
- **Fairy-SF extinction threats.**
  - For each extinction piece type: `denom = max(count_with_hand(Them, pt) - extinction_piece_count(), 1)`, then `score += make_score(1000,1000) / (denom*denom) * popcount(attacked & pieces(Them, pt))`.
  - A threat on the *last* piece is worth 4× a threat when 2 remain, and 9× a threat when 3 remain.
  - For pseudo-royal extinction pieces: `danger = 20 * attacks / (evasions + 1) + 40 * explosions; score += make_score(danger * (100 + danger), 0)`. This is quadratic in danger.
  - [Fairy-Stockfish evaluate.cpp](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/master/src/evaluate.cpp)
- **Fairy-SF extinction value per piece.** It adds `1000000 / (500 + PieceValue[MG][pt])`, so *cheaper* pieces count for more when the count is what matters. It also disables the "initiative" bonus for extinction variants, "to protect them from misidentification as drawish". — [Fairy-Stockfish evaluate.cpp](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/master/src/evaluate.cpp)
- **Fairy-SF rule options.** `extinctionPseudoRoyal: treat the last extinction piece like a royal piece`; `extinctionPieceCount: piece count at which the game is decided`; `extinctionOpponentPieceCount: opponent piece count required to adjudicate`; `extinctionClaim: extinction … can only be claimed as side to move`. — [Fairy-Stockfish variants.ini](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/master/src/variants.ini)
- **Fairy-Max does not know the baring rule.** It "is not aware of the shatranj rule that a bare king loses". A separate engine, ShaMax, was made because the rule changes play. — [fairymax(6) man page](https://manpages.ubuntu.com/manpages/noble/man6/maxqi.6.html)
- **The variants.ini I fetched has no `[shatranj]` block and no dedicated bare-king key.** Only the makruk/cambodian/asean `countingRule` exists. So I could not cite a Fairy-SF shatranj configuration. — [Fairy-Stockfish variants.ini](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/master/src/variants.ini)
- **Stockfish material-keyed endgame evaluators.**
  - `Endgames::init()` registers specialised evaluators by material key, e.g. `add<KPK>("KPK")` and `add<KRKP>("KRKP")`, in value and scale-factor maps.
  - The KXK evaluator: `material + push_to_edge(weakKing) + push_close(strongKing, weakKing)`, where `push_to_edge = 90 - (7*fd*fd/2 + 7*rd*rd/2)` (27 in the centre, 90 in the corners) and `push_close = 140 - 20*distance`. It adds `VALUE_KNOWN_WIN` when the material is known to win.
  - [Stockfish sf_15 endgame.cpp](https://github.com/official-stockfish/Stockfish/blob/sf_15/src/endgame.cpp)

### Inferences
- **Pseudo-royal lone guard.**
  - Under our bodyguard rule, an attack on the Kernel kills the cheapest adjacent guard. When `n_own_nonK == 1`, losing that guard leads into the bare-Kernel rule. In the 10773/11686/12113 pattern that is a forced loss, which the review shows at a40.
  - So the guard is pseudo-royal. Score it with the Fairy-SF king-danger form: `guard_danger = 20*A/(E+1)`, value penalty ∝ `guard_danger*(100+guard_danger)` (normalised), where:
    - A = the number of enemy shots or move-captures that reach the guard within our *2-action* window;
    - E = the guard's safe squares.
  - v4f has `hang2`/`reach2` as linear terms. The new parts are the quadratic shape and the "evasions in the denominator".
- **1/denom² instead of a hard gate.**
  - Rather than `lowmat ∈ {0,1}`, multiply the guard-threat feature by `1/max(n_own_nonK,1)²`. This weights threats 1.0 / 0.25 / 0.11 for 1 / 2 / 3 remaining pieces. It also covers the "one extra piece" endgames that the review's W1 seed list includes.
  - For parity-safe gating, keep it zero when n ≥ 3.
- **Material-key dispatch is the cleanest wiring for a late change.**
  - Keep `tanh(v_h + dv)` and add `lowmat * (w · f_endgame)` *inside the tanh*, the way Stockfish adds endgame.cpp terms on top of the main eval.
  - Positions with more material keep bit-exact parity with az_res4f_v2, so the gate only has to check endgames.
  - The KXK analogue for K+X vs bare K is `push_to_edge` → *push to the outer, about-to-lock ring* (our tie rule). The `push_close` analogue is negative for us, because Kernels within Chebyshev 2 trigger the side-that-just-moved rule. The closeness term must be parity-signed, not monotone.

### Gaps
- I could not find a primary source for how Shatranj-specific engines (ShaMax) evaluate the baring race, only that they implement the rule.
- I found nothing on Makruk counting-rule evals beyond the option's existence. Since our game has no draws, counting rules are likely irrelevant.

## Q3. Ludii / General Game Playing: feature-biased MCTS priors, and does it transfer to our prior?

### Takeaway
Ludii biases PUCT with a **linear softmax over a few action-anchored local-pattern features**, trained by cross-entropy against MCTS visit distributions. After pruning only ~15 features remain, and a few hundred self-play games are enough.

It transfers as a **logit correction on our policy** for the two prior errors the review lists: Kernel strikes from a recapturable square, and the flat warmup prior. That needs a graph change and 800-sim visit data, so it is post-contest.

### Cited Findings
- **Features and policy.**
  - A feature is a pattern of elements at relative positions: off-board, empty, friendly, enemy, piece n. Positions are "walks" of relative turns from an anchor.
  - Each feature carries the action it recommends: a from and a to relative position. Chess-like games need both.
  - The policy is `softmax(θᵀφ(s,a))` over the legal actions.
  - [Soemers, Piette, Browne 2019, ar5iv](https://ar5iv.labs.arxiv.org/html/1903.08942)
- **Training.**
  - Expert iteration with target π(s,a) = N(s,a)/ΣN, cross-entropy + L2 (λ = 1e-6), SGD α = 0.05.
  - One update per move, from a batch of 20 drawn from a 200-sample buffer.
  - Selection is PUCT with C = √2 using this prior. The first playout move is sampled from the apprentice.
  - [ar5iv 1903.08942](https://ar5iv.labs.arxiv.org/html/1903.08942)
- **Feature discovery and pruning.**
  - One new feature per self-play game, made by combining two co-active features, scored by `|r_err| × (1 − |r_combined|)`.
  - After 200 games the 15 features with the largest |θ| are kept, then 100 more games of weight-only training follow.
  - Simple combine-random strategies "already perform well".
  - [ar5iv 1903.08942](https://ar5iv.labs.arxiv.org/html/1903.08942)
- **Results.**
  - Significant gains vs UCT in Breakthrough, Fanorona, Hex 11×11, Othello and Teeko, after up to 200-300 self-play games.
  - A greedy linear policy with *no search* reached 40% vs UCT in Breakthrough and Teeko.
  - Only Yavalath got weaker.
  - [ar5iv 1903.08942](https://ar5iv.labs.arxiv.org/html/1903.08942); [arXiv abstract](https://arxiv.org/abs/1903.08942)
- **Follow-up work** aims at features over arbitrary geometries and at transfer and explanation of learned features. — [Browne, Soemers, Piette 2021, Strategic Features for General Games](https://arxiv.org/pdf/2101.00843)

### Inferences
- **Concrete transfer.** About 10-15 binary action features φ(s,a), computed in the graph from the action's from/to squares, added to the policy logits as `p_logit += θ·φ`. Candidates:
  1. The landing (or shooter's) square is adjacent to the enemy Kernel or inside its 2-action reach: the recapture trap, review (b) and W4.
  2. The move leaves the lone guard inside an enemy shot line.
  3. A Kernel move toward a lower-depth ring while `lowmat`.
  4. A capture that trades into `lowmat` while our ring depth ≥ theirs.
  5. A mover that is the only guard.
- **Fit.** θ is fit by cross-entropy against **800-sim visit distributions**: the review's 175+ live decisions plus self-play positions. That is 15 parameters, so it trains in seconds.
- **Why a linear head and not just more residual policy training.** The residual already had these positions and still put prior 0.82 on the 11362 a26 strike. Hand-anchored action features make the pattern explicit at the move level, which is the Ludii point. The team's v4f "guard-aware capture prior" is a single feature of this kind; the Ludii method generalises it and fits the weights from visits.
- **At 100 referee sims the prior ranks the moves** (earlier report: "Prior rank, not value sharpness, picks moves at 100 simulations"). Prior corrections therefore have direct leverage, but only on the specific patterns listed.

### Gaps
- The Ludii papers report results only as figures. I found no exact Breakthrough win-rate numbers beyond the 40% greedy figure.
- I found no evidence on whether Ludii-style features help *on top of* a trained CNN policy. They were benchmarked against plain UCT, not against an NN prior.

## Q4. Small-board AZ / MCTS games (Breakthrough, Onitama, Santorini): endgame handling and search+heuristic hybrids

### Takeaway
- **Implicit minimax backups** (Breakthrough, LOA, Kalah) are the best-documented way to inject an exact or heuristic tactical eval into MCTS. The referee search is fixed, but **our own self-play/teacher search is not**: use them to make better training targets.
- For endgames, the pattern from Lc0 is **rescoring training labels with exact endgame results**. Onitama work shows full retrograde solving of a whole small game is infeasible, but symmetry and forward-looking retrograde shrink it. A material-restricted tablebase (K+≤1 vs K+≤1) is the tractable analogue for us.

### Cited Findings
- **Implicit minimax backups.**
  - Keep win-rate estimates and heuristic evaluations separately, back up the heuristic values minimax-style, and use a blend to guide selection.
  - It "increases the strength of MCTS significantly" in Breakthrough, Kalah and Lines of Action, "even with simple evaluation functions".
  - α of 0.15-0.4 is a generally safe choice; in Breakthrough the best was about 0.5-0.6 under some settings (node priors, lower time settings).
  - [Lanctot, Winands, Pepels, Sturtevant 2014](https://arxiv.org/abs/1406.0486); also summarised in the [Maastricht MCTS-hybrids survey](https://dke.maastrichtuniversity.nl/m.winands/documents/mctshybrids.pdf)
- **Onitama (5×5).**
  - Strongly solving it by standard retrograde analysis is shown infeasible from state-count bounds.
  - A "forward-looking retrograde analysis" weakly solves it for the non-losing side.
  - A board symmetry halves the state space.
  - This is a 2025 Leiden bachelor thesis, not peer-reviewed. — [Leiden student thesis](https://studenttheses.universiteitleiden.nl/access/item%3A4262355/view)
- **Lc0 rescoring.**
  - When a training game reaches a Syzygy tablebase position, the rescorer rewrites the game result to the tablebase result, "so the network learns from the tablebase verdict rather than the game's mistakes".
  - Rescoring "is used in the training of all the best nets for leela, and should always be used before running training".
  - [lc0 Colab notebooks (chanind), via science.ecosyste.ms](https://science.ecosyste.ms/projects/175905); [CPW: Leela Chess Zero](https://chessprogramming.org/Leela_Chess_Zero)
- **Santorini AZ (cestpasphoto fork).**
  - The author targets "a smaller network but more MCTS simulations allowing to see further", and a curriculum from small/fast to larger nets. cpuct and FPU must be tuned per game.
  - Reported 90+% vs the Ai Ai bot and 95+% vs BoardSpace AI.
  - Its technical-details file (`README_features.md`) was not reachable (404 at the path I tried).
  - [cestpasphoto/alpha-zero-general](https://github.com/cestpasphoto/alpha-zero-general)

### Inferences
- **Rescoring for us (the Lc0 analogue).**
  - The endgame-start games in az_res4f_v2 were played at 100-200 sims *with* sampling, so their z labels include blunders in exactly the K+1 vs K+1 positions we want to learn.
  - Rescore every `lowmat` position in the saved data with a short **exact solver**: full-width over our 2-action turns, depth 1-2 turns, using the closed-form bare-Kernel rule and the firewall burn as exact terminals.
  - Replace z where the solver proves a win or loss, and keep z otherwise. With ~20 legal actions per action, one turn is about 400 action pairs, so depth 2 is about 160k leaves per position in the worst case. That is too slow in pure Python for all positions, but fine for the 268 live-type positions and a few thousand sampled ones with 1-turn depth plus proven-loss pruning.
  - This is also the only label source that is not biased the way root-Q is (the review notes Q is biased exactly here).
- **Implicit minimax in the teacher.** Run self-play/teacher MCTS with α ≈ 0.3-0.5 minimax backups of the hand_v4f eval plus exact bare-rule terminals. Forced lone-guard-loss lines then propagate up faster than averaging does. The review notes the 800-sim search found the loss "only after it is forced", which is the averaging-lag problem implicit minimax targets.
- **A tablebase for K+X vs K+Y (post-contest).**
  - Five non-Kernel types give 25 pairings, plus 5 for K+X vs K.
  - On the board open after round 8, with N open squares: N⁴ placements / 8 (D4 symmetry; the rings are concentric, so the board is D4-symmetric even though colours are not) × 2 sides × 2 action-index states.
  - The game is **finite-horizon** (the board locks at rounds 24/26), so it is plain backward induction over rounds, not iterative retrograde. Each round layer depends only on the next.
  - N = 36 gives about 0.84M states per pairing per round layer, and about 21M across all pairings. That is feasible offline in vectorised numpy, not in the remaining contest time.
  - Uses: (a) Lc0-style rescoring of all lowmat training labels; (b) training a value head directly on tablebase positions; (c) for K+X vs K (N³ ≈ 47k per type), possibly an exact Gather lookup table in the ONNX, if the initializer *size* limit allows it. The cap we know is a count of 512.
- "Smaller network, more sims" (Santorini) does not apply: our sim count is fixed at 100. It does support the team's choice to keep the trunk small and the evaluator fast.

### Gaps
- I could not confirm the exact Lc0 rescorer mechanics: whether it replaces only z or also the policy targets, and its "deblunder" option. I believe the rescorer has a deblunder mode that rewrites z from the first temperature-sampled blunder using root Q, but I found no source in this session, so treat that as unverified.
- I found no Onitama AZ project write-up that describes an endgame treatment.
- The Haque et al. "Road to Perfection" paper (Lc0 vs tablebases) returned 404, so I have no numbers on how often an AZ-style net misplays simple endgames.

## Q5. Bot-programming contests (CodinGame, Generals.io): evaluation design and endgames

### Takeaway
I found **little transferable material**. The CodinGame top write-ups I reached use MCTS or decoupled-PUCT with hand evaluations or random rollouts, not value networks. The one NN/RL self-play account reports a **passivity collapse**, which echoes our 11791 shuffling loss.

### Cited Findings
- **Spring Challenge 2021.** NN/RL bots on CodinGame date to 2019 (pb4, Agade). An AlphaZero-inspired pipeline "would often collapse into both sides WAITing most of the rounds", and a single-player MCTS that predicted the opponent was "an awful direction". — [CodinGame forum, Spring Challenge 2021 strategies](https://forum.codingame.com/t/spring-challenge-2021-feedbacks-strategies/190849?page=4)
- **Summer Challenge 2024, Legend 5th.** Decoupled PUCT without chance nodes, a layered tree per player, only 4 random moves simulated, then a heuristic positional score. No value net. — [CodinGame forum, Summer Challenge 2024](https://forum.codingame.com/t/summer-challenge-2024-feedback-and-strategies/204471/14)
- **Summer Challenge 2025, 5th.** Multi-tree MCTS for movement and heuristics for combat. — [CodinGame forum, Summer Challenge 2025 post-mortem](https://forum.codingame.com/t/summer-challenge-2025-post-mortem/207074)

### Inferences
- **Passivity collapse.** The 11791 loss (the Trace shuffling F7↔F6 on the ring that will lock) is the same failure mode. A cheap guard is a `lowmat × guard_on_next_lock_ring` feature: the review's third feature, which matches the race term of Q1 with a "rounds until lock" multiplier.
- I found no CodinGame or Generals.io write-up on shooting-unit endgames or shrinking maps with a learned evaluator.

### Gaps
- I did not reach the pb4/Agade NN write-ups, Ultimate Tic-Tac-Toe NN bot post-mortems or any Generals.io bot write-ups in this session. Their content is unverified.

## Q6. What transfers, ranked by expected gain per hour (before 16:20 vs post-contest)

### Takeaway
Only **#1** fits before the 16:20 upload, and only if someone can start it immediately (about 35-45 min including a 200-game gate). Otherwise **keep az_res4f_v2 live**: it already captured the cheap endgame-data gain. Everything else is post-contest.

### Cited Findings
- Each item's evidence is cited in Q1-Q4 above.
- The team's measured state: az_res4f_v2 is 76% vs az_res_v20 and 82% vs az_res_v6. Endgame MSE is 0.340 / 0.320 (MAG / CYAN). The graph has 278 nodes and 321 initializers, at 1.27 ms per batch-8 call. — `agents/az/RESULTS.md` (local)

### Inferences
**1. [FITS BEFORE 16:20 if started now] Material-keyed additive endgame term inside the tanh.**
- Sources: Stockfish endgame.cpp dispatch, the KOTH table shape, Fairy-SF pseudo-royal and 1/denom².
- **Graph:** `v = tanh(v_h + dv + lowmat * (w · f))`. `lowmat` is 1 iff each side has ≤ 1 non-Kernel piece, so positions with more material keep exact parity with az_res4f_v2.
- **f** (about 7 numbers):
  - one-hot of `d = ring(ownK) − ring(enK)` on the currently open board, buckets {≤−2, −1, 0, +1, ≥+2}. Bucket 0 is multiplied by +1 if we are MAGENTA and −1 if CYAN. Optionally add +1 to the Kernel's ring index for each enemy shot covering its inward step (the KOTH attacker increment).
  - `guard_danger = A/(E+1)` and its square, for the lone guard. A = the number of enemy 2-action attacks on the guard, E = its safe squares.
  - `bare_in_reach` = our `n_nonK == 0` and the enemy has a piece. v4f already has `bare` features, so skip this one if it is already there.
- **Fit:** logistic or MSE regression of z on f, with offset `v_h + dv` frozen (only w is free), on the 3840 held-out endgame-start positions plus the 268 live low-material positions. Use L2 and clip |w|.
  - Check that MAG-lost mean v goes from −0.50 toward −0.7, without moving won positions.
- **Gate:** 200 games vs az_res4f_v2, plus the validator. Upload only if Wilson LB > 0.5. In practice, also accept "no worse plus the endgame calibration fixed", because most games never reach `lowmat`.
- **Risk:** feature-plane bugs in the new ONNX nodes. Mitigate with the step-0 parity check on non-lowmat positions, and by eyeballing f on 10773/11686/12113 a35-a40.

**2. [POST-CONTEST, highest long-run value] Exact-solver / tablebase rescoring of low-material labels (the Lc0 analogue).**
- Start with a 1-2-turn full-width solver over lowmat positions in the saved data. Replace z when proven. Retrain the residual on the v20 recipe.
- Then build the finite-horizon K+X vs K+Y tablebase by backward induction over rounds, with D4 symmetry and about 21M states per round layer. Use it for rescoring and as direct value-head training data.

**3. [POST-CONTEST] Ludii-style linear action-feature prior correction.**
- 10-15 hand-anchored action features (recapture square adjacent to the enemy Kernel, lone guard into a shot line, Kernel toward the outer ring in lowmat, trade into lowmat while worse on ring depth), with `p_logit += θ·φ`.
- Fit θ by cross-entropy against 800-sim visit distributions. This targets W4 / pattern (b) and part of W2.

**4. [POST-CONTEST] Implicit-minimax teacher search for self-play targets.**
- α ≈ 0.3-0.5 minimax backup of the hand_v4f eval plus exact bare-rule terminals in *our* self-play MCTS (the referee's search stays unchanged).
- This gives sharper Q targets in forced endgame lines, where averaging MCTS lags (the review's "800-sim finds the loss only after it is forced").

**5. [POST-CONTEST] An ONNX dilation-BFS "safe path to inner ring" plane.**
- The Fairy-SF "on hold" trick with MaxPool 3x3 iterations masked by enemy attacks.
- Replaces raw ring depth in #1 with the effective race distance. About 8-12 nodes.

**6. [Low priority] Phase-tapered race term.**
- Apply the ring/race term at all material levels, tapered by material (KOTH applies its bonus in every phase), instead of the hard `lowmat` gate.
- It breaks parity, so it needs a full gate. It may prevent the round 8-9 corner retreats *before* material gets low.

### Gaps
- None of these gains have been measured on our game. The ranking is by mechanism match to the six live losses and by implementation cost.
- #1's fit uses random endgame starts plus 268 correlated live positions, so it can overfit the live shapes. Keep the weights small and clipped.
- I do not know the exact ring geometry per round. N = 36 open squares after round 8 is an assumption for the tablebase size estimate.
