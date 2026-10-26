# King safety, threats/hanging pieces and Texel tuning: what transfers to the COLONY: ONE hand evaluator

Scope: classical (pre-NNUE) evaluation terms from Stockfish 11 and Ethereal, SEE and hanging-piece terms, royal-piece variants, Texel-tuning practice, and phase-tapered weights. Each item is mapped onto the `hand` evaluator (fixed features -> linear -> tanh, Texel-fitted on self-play outcomes) and the W2/W3/W4 weaknesses in `reviews/hand_v2.md`.

Local context used for the mapping (read from the repo, not web sources):
- `agents/hand/fit.py`: `fit_value` minimises BCE(2*lin, (z+1)/2) with LBFGS and `l2 * (w**2).sum()`. The penalty shrinks toward **zero**, not toward the hand-set start weights. The split is an 85/15 sequential cut. `fit4.py` then refits on all positions and prints "val acc" on the same rows, so the second number is in-sample.
- `agents/hand/gen_data.py`: stores every root position of each self-play game, with the outcome (±1, side-to-move view) and root visit counts. It does **not** store the root Q / search value.
- Data size: 600 v1 games gave 21.7k positions (~36 per game). So 53k positions is ~1.5k games and 84k is ~2.3k games, for roughly 100+ value features (61 `add` call sites in `hand.py`, several looped over 5 piece types).
- v4 already added `own_hang2_*`, `en_hang2a2_*`, `ownK_g1_t2`/`enK_g1_t2` and a constant guard-aware Kernel-attack prior (`ek_guard 1.5`). Results: v4f with a full refit scored 55% [48,61] vs v3. v4p, with the same terms hand-set at ±0.4 and no refit, scored 38% [32,45].
- Guard priority Ping -> Exploit -> Trace -> Daemon -> Ice equals **ascending piece value** (15, 25, 30, 45, 50). So "the priority guard dies" means "the cheapest adjacent guard dies".

---

## Q1. Stockfish classical king danger, threats and mobility: which non-linearities matter, and how do they map onto the bodyguard mechanic?

### Takeaway
Stockfish's king safety is a **linear "danger" counter, passed through a thresholded quadratic** (`kingDanger²/4096` mg, `/16` eg, only if > 100). Defensive quality (pawn shelter) and the absence of the key attacker (queen) enter **inside** the counter, before the square. The threats term is mostly linear counts, gated by "weak", meaning attacked and not strongly protected. For us: the guard count plays the shelter's role and belongs inside a squared danger term, together with strikers. Stockfish's "safe check" distinction (2-7x the weight of an unsafe check) maps directly onto "the Kernel attacker's origin square is / is not attacked", which is exactly the W4 failure.

### Cited Findings
- **King ring and attacker counters (SF11):**
  - The king ring is the king's neighbours plus the king square, clamped to files B-G and ranks 2-7, minus squares double-defended by own pawns.
  - `kingAttackersCount` counts pieces whose attacks touch the ring.
  - `kingAttackersWeight` adds `KingAttackWeights = {0,0,81,52,44,10}` (none/pawn/knight/bishop/rook/queen).
  - `kingAttacksCount` counts attacks on squares adjacent to the king.
  - — [Stockfish 11 evaluate.cpp](https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_11/src/evaluate.cpp)
- **Safe checks (SF11):** QueenSafeCheck 780, RookSafeCheck 1080, BishopSafeCheck 635, KnightSafeCheck 790. Unsafe checks only add `148 * popcount(unsafeChecks)`. — [Stockfish 11 evaluate.cpp](https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_11/src/evaluate.cpp)
- **kingDanger formula (SF11):**
  - Positive terms: `kingAttackersCount*kingAttackersWeight`, `+185*popcount(kingRing & weak)`, `+148*unsafeChecks`, `+98*popcount(blockers_for_king)` (pinned/discovered blockers), `+69*kingAttacksCount`, `+3*kingFlankAttack²/8`, `+mg(mobility[Them]-mobility[Us])`.
  - Negative terms: `-873*!(enemy queen)`, `-100*(knight defends king area)`, `-6*mg(score)/8` (score initialised from the pawn shelter, `pe->king_safety`), `-4*kingFlankDefense`, then `+37`.
  - Transform: **if kingDanger > 100**, subtract `kingDanger²/4096` (mg) and `kingDanger/16` (eg).
  - — [Stockfish 11 evaluate.cpp](https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_11/src/evaluate.cpp)
- **Weak squares (SF11):** enemy-attacked squares that we do not attack twice and that are either unattacked by us or defended only by our king/queen. — [Stockfish 11 evaluate.cpp](https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_11/src/evaluate.cpp)
- **Threats (SF11):**
  - Definitions: `stronglyProtected = attackedBy[Them][PAWN] | (attackedBy2[Them] & ~attackedBy2[Us])`; `weak = pieces(Them) & ~stronglyProtected & attackedBy[Us]`.
  - **Hanging S(69,36)** applies to weak pieces that are either undefended, or are non-pawns attacked twice by us.
  - ThreatByMinor by victim (P,N,B,R,Q): S(6,32), S(59,41), S(79,56), S(90,119), S(79,161).
  - ThreatByRook by victim: S(3,44), S(38,71), S(38,61), S(0,38), S(51,38).
  - **ThreatBySafePawn S(173,94)** is the largest threat term: non-pawns attacked by pawns that are themselves safe.
  - Others: ThreatByPawnPush S(48,39), ThreatByKing S(24,89), RestrictedPiece S(7,7), WeakQueen S(49,15), SliderOnQueen S(59,18), KnightOnQueen S(16,12).
  - — [Stockfish 11 evaluate.cpp](https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_11/src/evaluate.cpp)
- **Mobility (SF11):** per-piece tables indexed by the number of attacked squares in the mobility area. The area excludes own blocked/low-rank pawns, own king/queen, king blockers and **squares attacked by enemy pawns** (safe mobility). Ranges: knight S(-62,-81)..S(33,33), queen S(-39,-36)..S(116,212). — [Stockfish 11 evaluate.cpp](https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_11/src/evaluate.cpp)
- **Ethereal's analogous non-linearity:** a safety counter S is split into mg/eg, then the eval adds `-mg*max(0,mg)/720` (mg) and `-max(0,eg)/20` (eg). The eg is linear and the mg a one-sided square. — [TalkChess: king safety scaling](https://talkchess.com/viewtopic.php?p=902040); same functions written as `fmg(x) = -x·max(0,x)/720` in [Grant, Evaluation & Tuning in Chess Engines](https://github.com/AndyGrant/Ethereal/blob/master/Tuning.pdf)
- **Attacker-count gating (CPW):**
  - The sample king-zone scheme multiplies attack value by an `attackWeight[attackingPiecesCount]` table: 1 attacker -> 0, 2 -> 50, 3 -> 75, 4 -> 88, 5 -> 94, 6 -> 97, 7 -> 99 (percent). One attacker contributes nothing.
  - Rebel and Stockfish index a "SafetyTable" by attack units (minor 2, rook 3, queen 5, +6 for a safe queen contact check). The curve is "S-shaped": slow, then fast, then flat.
  - — [CPW King Safety](https://www.chessprogramming.org/King_Safety)
- **Forum practice:**
  - jdart passes summed king-safety terms through a sigmoid lookup table whose parameters are Texel-tuned ("tricky because the gradient is non-linear").
  - No4b reports roughly 40-50 Elo from the attacker-count table, and found putting all safety in the opening phase best in Drofa.
  - These are forum reports, not controlled studies.
  - — [TalkChess: king safety scaling](https://talkchess.com/viewtopic.php?p=902040)

### Inferences
- **Guards are the shelter term and go inside the non-linearity.** In SF11 the shelter enters as `-6*mg(score)/8` before squaring, so shelter is worth more when the attack is heavy. Our analog:
  `dangerK_own = relu(a1*n_strike1_en + a2*n_strike2_en + a3*shooters_in_range_en + a4*weak_esc_own - b*g_own + c)²`.
  - `n_strike1` = enemy pieces that hit our Kernel square now. `n_strike2` = enemy pieces with a quiet move then a strike (the `t2` map, already blocking-aware).
  - `weak_esc_own` = Kernel-adjacent empty squares hit by the enemy and not in the next firewall ring.
  - The CPW "1 attacker -> 0" table is the same idea: with g guards, the Kernel only falls if strikes available in 2 actions exceed g. That is a threshold/quadratic in (strikes - guards), and today's linear `t2_own` (fitted 0.001) cannot express it.
- **Shooters are the "queen" of this game.** SF zeroes most danger without the enemy queen (`-873`). Here Trace (diagonal shot 2-3) and Daemon (adjacent shot) strike without moving, so the same shooter can strike twice in one 2-action turn (guard, then Kernel) and is never recaptured on the target square. A multiplicative gate like `dangerK * (shooter_alive_en)`, or a separate weight for shooter strikes vs move strikes, is the direct analog.
- **Safe check -> safe Kernel strike (fixes W4).** A strike on a guarded Kernel kills the cheapest guard and leaves the attacker on its **origin** square. So the real exchange is `V(cheapest guard) - V(attacker) * hit_by_defender(origin)`. In 6032 a24 the Ice (50) struck from a square where it was recaptured, killing EX (25): net -25, while the prior gave 0.98. SF's 780-1080 vs 148 safe/unsafe gap says this distinction carries most of the signal.
- **"Blockers for king" (98 per blocker) -> discovered-strike blockers.** Own pieces sitting on an enemy Trace diagonal or Daemon line to our Kernel are latent strikes: if they move, the Kernel is hit. Count = unblocked-reach strikes minus blocked-reach strikes, built from stopgap's `reach()` kernels. Medium effort; lower priority.
- **Mobility:** SF's safe-mobility area (excluding enemy-pawn-attacked squares) maps to "destination squares not in `hit_en` and not in the next burning ring". For the Kernel this is the existing `esc_own`. Piece mobility for non-Kernels is probably low value here: MCTS already searches moves, and the evaluator's weak spot is tactics.
- **Threat terms that transfer:**
  - **Hanging-if-undefended-or-double-attacked**: our 2-action analog is "attacked, and side-to-move has 2 actions" (`en_att2_*`, already present).
  - **Threat by a cheaper/safe attacker** (ThreatBySafePawn, the largest SF threat): `en_att_safe_<t>` = enemy t attacked by our piece whose own square is not hit by the enemy. New, cheap.

### Gaps
- No source states why SF's eg king danger is linear; the forum thread raises the question but has no answer.
- Pawn shelter/storm code lives in `pawns.cpp` (not fetched).
- No source quantifies the Elo of individual SF11 king-danger components.

---

## Q2. SEE and hanging-piece terms in a static eval: can a 1-2 exchange SEE be built from attack/defend maps with our piece values?

### Takeaway
Engines with quiescence search treat static hanging terms as a weak, often redundant nudge (SF Hanging is only S(69,36); several authors measured no gain). **That argument does not apply to us:** the referee's MCTS-100 calls the evaluator on raw tactical leaves with no qsearch. The evaluator has to carry the first exchange itself. A 1-ply SEE is buildable from per-type hit maps with elementwise min/max. The one twist is the **shot vs move-capture** distinction: a shooter never lands on the target, so a "defended" target is still lost for free.

### Cited Findings
- **SEE (CPW):**
  - Definition: SEE "examines the consequence of a series of exchanges on a single square after a given move".
  - Recursive form: `see(sq, side) = max(0, value(piece_just_captured) - see(sq, other))`, always capturing with the smallest attacker. The `max(0, ·)` lets a side stop capturing (stand pat).
  - "Some programs optimize the SEE function to only return a losing or equal/winning flag".
  - Uses are mostly move ordering and pruning, not evaluation.
  - — [CPW Static Exchange Evaluation](https://www.chessprogramming.org/Static_Exchange_Evaluation)
- **Hanging piece (CPW):** an attacked piece not defended by its own side. Loose (undefended) pieces may be scored even when not attacked, especially with more than one or two of them while the opponent has double-attack/fork means. — [CPW Hanging Piece](https://chessprogramming.org/Hanging_Piece)
- **Forum evidence that it is redundant with qsearch:**
  - Kempelen proposed a penalty for attacked+undefended pieces, halved when that side is to move, and measured no improvement.
  - Bob Hyatt: "definitely redundant" with quiesce and less accurate (ignores pins/overloads).
  - Sven Schüle: no improvement, and false positives/negatives (e.g. attacked by a cheaper piece, attackers outnumbering defenders).
  - — [TalkChess: Evaluating attacked pieces](https://talkchess.com/viewtopic.php?p=232166)
- **Counter-evidence on use in eval:** a search summary attributes to Don Dailey that Komodo adds attack bonuses that grow when several pieces are attacked, and that DiscoCheck has a symmetric term that rises roughly linearly with the number of hanging pieces. I could not open that thread to verify. — [CPW/TalkChess search results, unverified](https://talkchess.com/viewtopic.php?p=232166)
- **SF11's SEE-like static gating:**
  - `stronglyProtected` (pawn-defended, or defended twice and not attacked twice) suppresses most threat bonuses.
  - "Hanging" fires for undefended pieces or non-pawns attacked twice (`attackedBy2[Us]`).
  - ThreatByMinor/Rook values grow with victim value (e.g. minor-on-rook S(90,119)).
  - — [Stockfish 11 evaluate.cpp](https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_11/src/evaluate.cpp)

### Inferences
**Why it matters more here.** With alpha-beta + qsearch, the static eval only sees quiet nodes, so hanging terms are redundant (Hyatt). In our setting:
- MCTS-100 with FPU 0 expands only ~13-14 evaluator calls per search (hand_v2 review, section 1).
- Every leaf, including "a piece is en prise" leaves, is scored by our value directly.

So a static SEE is our qsearch substitute, and W2 is precisely a missing 2-ply SEE. Expect these terms to be worth far more than in chess engines.

**1-ply SEE sketch (fixed-conv/elementwise).** Values V = {P:15, X:25, T:30, D:45, I:50, K:100}; the code's VALS = [1, 3, 1.7, 2, 3] for P, D, X, T, I is the same scale ÷15.

1. **Per-type attack maps** (side A, type k, shape [B,1,8,8]): `hA_k`. They exist inside `hand.py` before being OR-ed into `hit_own`/`hit_en`. Split them into:
   - `shotA = clamp(hA_T_shot + hA_D_shot)`: Trace diagonal 2-3 and Daemon adjacent; the attacker stays on its origin.
   - `moveA_k`: Ping diagonal ≤2, Daemon distance 2-3, Exploit L, Ice ≤2, Kernel 1; the attacker lands on the target.
2. **Cheapest move-attacker on each square** (ascending V order P, X, T, D, I, K):
   `cheapA = Σ_k V_k * moveA_k * Π_{j<k}(1 - moveA_j)` (5 multiplies, elementwise).
3. **Defence of the target square:** `defB = hit_B` restricted to B's occupied squares. Hit maps include squares occupied by own pieces; `D = hit_own*hit_en` in `fast4.py` already relies on that.
4. **SEE-1 per square, gain for A capturing B's piece:**
   `see1 = max( Vb * shotA , Vb * anyMoveA - cheapA * defB , 0 )`, with `Vb = Σ_t V_t * B_t` and `anyMoveA = clamp(Σ_k moveA_k)`.
   - A shot is a free capture regardless of `defB`.
   - The existing `own_hang`/`en_hang` use `hit*(1-hit_other)` and miss "defended but shot" and "defended but attacked by a cheaper piece" (a Ping hitting a defended Ice is +35).
5. **Scalar features:**
   - `seeA_max = ReduceMax(see1 * B_occupied)`: best single capture. Use max, not sum: one action captures one piece.
   - `seeA_2nd` via a 5-element sorting network (min/max ops on the 5 per-type scalars `S(B_t*see1)`): the second capture available with 2 actions.
   - Features from side-to-move's view:
     - `own_see_max * a2`, `own_see_max * (1-a2)`.
     - `en_see_max * a2`, `en_see_max * (1-a2)`.
     - `en_see_2nd * a2`.
     - `own_see_2nd`: the opponent gets 2 actions next turn, and we can usually save only one piece per action.
   - Fit them jointly (see Q4). The SF-style split is also cheap: `en_shot_<t>`, `en_att_cheaper_<t>` (attacked by a strictly cheaper mover) and `en_att_safe_<t>` (attacker's own square not hit).

**2-action correction (where chess intuition breaks).**
- A side with 2 actions can capture with action 1 and then move the capturer away, or capture the recapturer, with action 2. So "defended" protects much less when the capturing side has `a2`.
- This argues for the a2-split features and for letting Texel learn the discount, rather than hard-coding chess SEE.
- The opponent always moves with 2 actions, so `own_*` threats should be measured against the 2-action reach `hit2` that v4 added (`own_hang2_*`). Extending SEE to `hit2` (cheapest attacker over 1-2 action reach) is the "2-exchange" version.

**Kernel as a SEE target (fixes W4 and part of W3).** Because every Kernel attack kills the cheapest adjacent guard and leaves the attacker on its origin square:
- `Vg_B = Σ_t V_t * guard_B_t * Π_{s<t}(1 - guard_B_s)` (priority order = value order; `guard_B_t = S(B_t * conv3x3(B_K))`, a 0/1 scalar per type).
- `seeK_A = max_k [ kA_k * ( g0_B*WIN + (1-g0_B) * (Vg_B - V_k * S(A_k * hit_B)) ) ]`, with `kA_k` = "type k of A strikes B's Kernel" (scalars; one piece per type, so S(A_k * hit_B) is just "is that piece's square hit").
- Use it in two places:
  - **Value:** add `seeK_en` / `seeK_own`, split by a2.
  - **Prior:** replace the constant `ek_guard 1.5` with a to-term on the enemy-Kernel square of `cap * (Vg_en/15)`, plus a from×to rank-1 outer product `-cap * ownV(from) * hit_en(from) ⊗ enK(to)`, like the existing `kfw` outer product. The 0.98 prior on a losing strike then drops below quiet alternatives.

### Gaps
- No source measures static-SEE eval terms inside a no-qsearch MCTS. The "matters more here" claim is an inference from the referee setup, not a cited result.
- I could not verify the Komodo/DiscoCheck multi-hanging claims beyond a search-snippet summary.
- The swap-list (iterative) SEE page was not fetched; for a 1-2 exchange version the recursive definition above is enough.

---

## Q3. Royal-piece variants (Xiangqi, Shogi, Tafl, minichess): lessons for a king protected by sacrificial bodyguards

### Takeaway
The transferable lesson is from Xiangqi: **defenders' value depends on the opponent's remaining attacking material**. A scheme of "attackers minus defenders with a non-linear bonus" is the community proposal, which is the same structure as the SF counter in Q1. Shogi's Bonanza captured king safety implicitly with **king-relative piece-square tables (KPP/KKP)** learned from data. That is too many parameters for our ~2k games except in a heavily bucketed "tropism" form. I found no usable sources on Tafl or minichess evaluation in the time available.

### Cited Findings
- **Xiangqi engine notes:** a king-safety term rewards advisors/elephants still at home and penalises an open file in front of the general. — [metacpan Game::Xiangqi::Engine (search-result summary; page returned HTTP 402)](https://metacpan.org/pod/Game::Xiangqi::Engine)
- **Xiangqi forum proposals:**
  - Score the attackers-minus-defenders difference with a non-linear bonus, so two horses already earn a bonus against a side with no advisors/elephants, and more attackers are needed as defenders are added.
  - Defender (advisor/elephant) value depends on the opponent's attacking material (e.g. rook vs rook is a draw with no defenders).
  - These are unverified forum proposals that partly disagree.
  - — [TalkChess xiangqi threads (search summary)](https://www.talkchess.com/forum/viewtopic.php?p=360763)
- **Shogi / Bonanza:**
  - Bonanza uses piece-square tables "indexed by king location and further two-piece locations and side to move" (KPP, KKP, KPPT). They were used in many shogi programs and "influenced the design of NNUE".
  - Learning: the "Bonanza Method", a supervised move-adaptation method that "evolved to Minimax Tree Optimization (MMTO)".
  - — [CPW Bonanza](https://www.chessprogramming.org/Bonanza)
- A forum poster notes that in shogi the king is typically chased all over the board, so static castle-based king tables matter less. — [TalkChess search result](https://talkchess.com/viewtopic.php?p=947065)

### Inferences
- **Bodyguard = consumable shelter, like advisors that die one per attack.** The Xiangqi "defender value depends on attacking material" lesson suggests interacting guard terms with enemy striker material: `g_own * shooters_alive_en`, `g_own * n_strike_any_en`. A guard is worth little if no enemy piece can reach the Kernel within 2 actions, and a lot when 2+ can.
- **Guard value is not uniform.** The cheapest guard always dies first, so the material at risk from one strike is `Vg` (15-50), not "a guard". A Kernel guarded only by an Ice is a 50-point liability to every striker. A Ping next to the Kernel is a cheap shield, the analog of an intact pawn shield. Features: `Vg_own` and `Vg_own * n_strike_en`.
- **Bonanza/KPP, scaled down = tropism by Chebyshev distance to the enemy Kernel.** Use per piece type × distance bucket {1, 2, 3, ≥4}: about 6×4 = 24 features per side via fixed 15×15 distance-bucket convs on the Kernel plane. It captures "Ice at distance 2 from the enemy Kernel gets eaten by a 2-step Kernel walk" (6032 a23) and "shooter at diagonal 2-3". Partly redundant with v4's `kz2` and `hang2`.
- **MMTO (fit the eval so that the best move's child outscores its siblings)** is conceptually the right objective for a value used in move selection, but it needs child evaluations for every sibling. Out of scope for the remaining time. The existing `fit_policy` on visit counts is the cheap cousin for the prior.

### Gaps
- No source found on Tafl (king captured by surrounding) or Gardner/Los Alamos minichess evaluation; not searched further because of the tool-call budget.
- No primary YaneuraOu/Apery pre-NNUE king-safety documentation was retrieved. The Bonanza page gives only the high-level structure.
- Xiangqi material is forum-level and partly contradictory.

---

## Q4. Texel-tuning best practice (quiet positions, labels, K, regularisation, dataset size, outcomes vs search scores, pitfalls)

### Takeaway
Our fit is already Texel with a cross-entropy loss and a free scale, which is fine. The highest-value fixes are about **data and priors, not the loss**:
- (a) Regularise toward the hand-set weights, not toward 0. Rare-but-decisive features like `t2_own` and `g1_t2` are being shrunk to ~0 because self-play rarely shows them.
- (b) Always refit jointly after adding a feature. Hand-setting on top of fitted correlated weights cost 17 points (v4p 38%).
- (c) Validate with a game-grouped split.
- (d) Next data generation: store the search value too and blend λ·outcome + (1-λ)·search, as NNUE trainers do.

The "quiet positions only" rule does **not** transfer directly: our evaluator is called at non-quiet MCTS leaves, so it must be trained on them too.

### Cited Findings
- **Texel's method (Österlund):**
  - Error `E = 1/N Σ (R_i - Sigmoid(q_i))²`, with R ∈ {0, 0.5, 1} and `q_i` the white-relative **quiescence** score.
  - K is chosen once to minimise E and "never changed again"; K = 1.13 for Texel 1.03.
  - Data: 64,000 games at 1s+0.08s, giving ~8.8M positions (~140 per game). Book positions and mate-score positions were excluded.
  - ~400 parameters; local search ±1 per parameter.
  - Cumulative +99.6 Elo over 7 sessions.
  - Österlund first removed positions where the qsearch score deviated from the game search score, then found **including** them better (the +39.4 Elo step).
  - Listed pitfalls: correlation is not causation; the engine "cannot learn what it doesn't understand"; dependence of the ~140 positions per game (argued acceptable with 64k games for ~400 parameters).
  - — [CPW Texel's Tuning Method](https://www.chessprogramming.org/Texel%27s_Tuning_Method)
- **Ethereal (Andrew Grant):**
  - "using a large number of positions from the same game has been shown to produce a lower quality dataset".
  - Early data: Zurichess (≈1.6M), Laser positions from **randomly exploring search trees** (≈1.3M), and Ethereal self-play (≈4.9M).
  - Current data: positions sampled from 1M hyper-bullet self-play games, then **resolved by applying the full PV of a high-depth search**, labelled with the original game's WDL. The paper notes "the increased diversity and the conversion of position to a quiet state, appears more important than the WDL result precision".
  - Sigmoid switched to base e with 1/400 absorbed into K (`σ = 1/(1+e^{-KE})`). MSE loss; L1 tried "with no success".
  - Fischer Random positions mixed in for diversity: "equal or better results".
  - Non-linear king safety is tuned by gradient, with explicit derivatives through `fmg(x) = -x·max(0,x)/720`.
  - Optimiser: AdaGrad, currently full-batch.
  - — [Grant, Evaluation & Tuning in Chess Engines (Tuning.pdf)](https://github.com/AndyGrant/Ethereal/blob/master/Tuning.pdf)
- **Outcome vs search-score labels in NNUE training:**
  - nnue-pytorch's `lambda_` = 0.0 trains purely on game results, 1.0 purely on search scores, and in between interpolates. The target is `λ·sigmoid(eval/scale) + (1-λ)·result`.
  - A forum post recalls Stockfish training at λ = 1.0 (scores only); unverified.
  - Suggested workflow: train on a previous engine's evals, then fine-tune with low λ.
  - — [nnue-pytorch lightning_module (mirror)](https://isip.piconepress.com/projects/ai_chess/resources/pro_player_models/nnue-pytorch/model/lightning_module.py); [CPW NNUE](https://www.chessprogramming.org/NNUE); [TalkChess](https://talkchess.com/viewtopic.php?p=941460)
- **Tuning non-linear king safety:** jdart Texel-tunes the parameters of a sigmoid king-safety table and calls it "tricky because the gradient is non-linear". — [TalkChess: king safety scaling](https://talkchess.com/viewtopic.php?p=902040)

### Inferences (concrete changes to `fit.py` / data)
1. **Prior-centred ridge (10 min, likely the best gain per hour).**
   - Replace `l2 * (w**2).sum()` with `l2 * ((w - w0)**2 * r).sum()`, where `w0` = the hand-set or previous weights and `r` = per-feature strength, lower for well-supported features.
   - Why: `t2_own` → 0.001 and `t2_en` → -0.03 (hand_v2 review W3) is the textbook symptom of a rare feature in self-play data. The generating policy avoids lone-guard-under-strike positions, so the outcome signal is tiny, and zero-centred L2 plus correlated neighbours (`ownK_thr`, `g0`, `g1`) absorb it.
   - This is Österlund's "cannot learn what it doesn't understand", amplified by small data: ~1.5-2.3k games for ~100+ weights, against Texel's 64k games for ~400 parameters (16 vs 160 games per parameter).
2. **Never hand-set on top of fitted weights; refit jointly.** v4p (hand-set ±0.4, no refit) scored 38%, while v4f (refit) scored 55%. Correlated features double-count when hand-set. Combine with (1): put the hand guess in `w0`, then refit.
3. **Game-grouped validation and honest reporting.** The current cut is sequential, so it is nearly game-grouped. But `fit4.py`'s second fit runs on all rows and prints val acc on rows it trained on, so "0.752" may be optimistic. Compare models only on the first-fit val number, or better, on arena results.
4. **"Quiet positions" adapted to MCTS.**
   - Texel and Ethereal resolve to quiet positions because the alpha-beta eval is only called at quiet qsearch leaves. Our evaluator is called at every MCTS leaf, including mid-exchange positions and actions_left = 1 mid-turn positions. **Keep non-quiet positions** (Österlund's +39.4 Elo came from re-including them) and instead add SEE features (Q2) so the linear model can explain them.
   - Ethereal's "random search-tree positions" (Laser) suggests **augmenting with off-path MCTS leaves** (e.g. children of the 800-sim roots the `prior` direction already generates). These contain the blunder positions (lone guard under strike, Ice next to the enemy Kernel) that greedy self-play never reaches, which is exactly what W2/W3 need. They have no game outcome, so label them with the search value (λ = 1 for those rows).
5. **λ-blend outcome and search value (next data generation).**
   - `gen_data.py` stores visit counts but not the root Q. Add the root value (one line), and fit to `y = λ·(q+1)/2 + (1-λ)·(z+1)/2`.
   - Labels from a 100-sim search on the same evaluator inherit its blind spots (the hand_v2 review notes 6032 a36 shows only dQ 0.14 at 800 sims). So use 800-sim Q if available, and keep λ moderate.
   - I have no source for an optimal λ in small-data self-play; 0.3-0.5 is a guess to A/B.
6. **K / scale.** With free linear weights, K is absorbed (Ethereal absorbs 1/400 into K). The tanh scale only matters for MCTS calibration. The team's "all value ×1.3 → 49%" test is consistent with the Texel scale already being near-calibrated, so leave it.
7. **Fit the non-linear king-danger term jointly with autograd.** `lin = Fe@w + b - c_own*relu(Sd_own@wd + bd)² + c_en*relu(Sd_en@wd + bd)²`. Ethereal does exactly this with derivatives through the max/square. LBFGS in `fit.py` handles it with ~15 lines of change. Initialise `wd` from hand guesses (striker counts positive, guards negative), and centre the ridge on them.
8. **Optional: drop or down-weight warmup positions** (the first 8 actions, sampled with temperature). This mirrors Texel's exclusion of book positions. Expected effect is small.
9. **Diversity:** Ethereal found Fischer-Random mixing "equal or better". Our analog is mixing games from several generators (v1/v2/v3/az_res/distill opponents) rather than pure self-play, to reduce self-play bias.

### Gaps
- No source gives Texel/Ethereal regularisation strengths. The CPW page only links to forum threads on "regularization and cross-entropy".
- No source on Texel tuning for evaluators used inside MCTS (vs alpha-beta).
- The exact Texel sigmoid form is an image on CPW, so the `10^(-Kq/400)` form is from general knowledge, not verified here.

---

## Q5. Tapered / phase-dependent weights: interpolate by firewall phase / round and by actions_left?

### Takeaway
Yes, selectively. Classical engines interpolate every term between mg and eg by a material phase. Ethereal's phase is linear in material (Fruit-style). Its king safety is quadratic in mg and linear/absent in eg, and one engine found safety best as opening-only. For us, the natural phase is **round progress u plus locked rings**, and **actions_left (a2)** is a second, more important axis for every threat/SEE term. Implement it as feature × phase products in the linear model, restricted to a small subset, with the prior-centred ridge.

### Cited Findings
- **SF11 final interpolation:** `v = (mg*phase + eg*(PHASE_MIDGAME - phase)*sf/SCALE_FACTOR_NORMAL) / PHASE_MIDGAME`, then sign flip for black and a Tempo bonus. — [Stockfish 11 evaluate.cpp](https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_11/src/evaluate.cpp)
- **Ethereal phase:**
  - Fruit-derived, `λ = [1,1,2,4]·[N,B,R,Q]`, scaled to `ρ` on 0..256. Final eval `((256-ρ)E_mg + ρξE_eg)/256` with an endgame scale ξ.
  - Gradients for mg/eg weights are the same feature coefficient multiplied by `ρ_mg = 256-ρ` or `ρ_eg = ξρ`. This shows tapering is just "feature × phase" columns in a linear fit.
  - — [Grant, Tuning.pdf](https://github.com/AndyGrant/Ethereal/blob/master/Tuning.pdf)
- **King safety by phase:** SF eg king danger is `kingDanger/16` (linear) vs mg `kingDanger²/4096`, and Ethereal's eg is `-max(0,eg)/20`. In Drofa, "putting the whole safety score in the opening phase worked best" (forum report). — [TalkChess: king safety scaling](https://talkchess.com/viewtopic.php?p=902040)

### Inferences
- **Phase variable:**
  - Firewall rounds give a hard schedule. The board shrinks at the end of rounds 8/16/24/26, so the "endgame" is driven by ring locks, not only material.
  - `hand.py` already has `u`, `u1`, `u2`, `nlock`, `mat_x_nlock`, `mag_late`.
  - A cheap tapered version: `ph = clamp(nlock/3, 0, 1)`, or `ph = 0.5*u_round + 0.5*(1 - material/total_material)`. Add `f*ph` columns for:
    - material difference;
    - the SEE max terms (Q2);
    - Kernel danger (Q1), whose damage grows as the board shrinks and escape squares vanish;
    - Kernel ring/centrality.
  - Keep the rest un-tapered. That is about 8-12 new columns, which ~2k games can support with the prior-centred ridge.
- **actions_left is the more important axis.** Every threat term has a different meaning with 1 vs 2 actions left (capture + escape, double strike on a lone guard). The existing `_a1`/`a2` splits (`ownK_thr_a1`, `en_att2_*`, `en_hang2a2_*`) are the right pattern. Apply it to every new SEE / Kernel-danger feature rather than tapering by round.
- **Bare-Kernel ring rule** (the smaller ring wins, ties go to MAGENTA) makes the eg Kernel term about rings, not safety. This matches SF's choice of making eg king danger nearly linear: keep the quadratic danger term un-tapered or gated by `(1 - bare_K)`.

### Gaps
- No source on phase definitions for games whose board shrinks; the firewall phase choice is our inference.
- Whether tapering helps at only ~2k games is untested. Tapering doubles parameters for the chosen subset, so keep the subset small.

---

## Q6. Ranked recommendations (expected gain per hour, within ~1 h each, ONNX-friendly)

### Takeaway
Do fitting fixes first (minutes, zero ONNX risk), then a Kernel-aware SEE (cheapest-guard value + attacker-origin safety) shared by value and prior, then the squared Kernel-danger term. Everything is fixed convs, elementwise min/max/mul, ReduceSum/ReduceMax and rank-1 outer products. Every new feature should be **refit jointly** and arena-tested (≥200 games, as in the v3/v4 A/Bs).

### Cited Findings
- SF11's safe-vs-unsafe check weighting (780-1080 vs 148), the shelter-inside-the-quadratic structure and the ThreatBySafePawn dominance — [Stockfish 11 evaluate.cpp](https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_11/src/evaluate.cpp)
- The CPW attacker-count table (1 attacker = 0) — [CPW King Safety](https://www.chessprogramming.org/King_Safety)
- Ethereal's joint gradient tuning of non-linear safety, and its search-tree/PV-resolved data — [Grant, Tuning.pdf](https://github.com/AndyGrant/Ethereal/blob/master/Tuning.pdf)
- Texel's finding that including non-quiet positions helped — [CPW Texel's Tuning Method](https://www.chessprogramming.org/Texel%27s_Tuning_Method)
- NNUE λ blending — [CPW NNUE](https://www.chessprogramming.org/NNUE)

### Inferences (ranked; times are estimates)
1. **Fit hygiene** (~15 min, low risk, applies to every later refit):
   - Prior-centred ridge `l2*||w - w0||²`, with `w0` = the hand guesses for new/rare features (g1_t2 ±0.4, hang2 ≈ -0.35·own_X).
   - Joint refit.
   - Honest game-grouped val.
   - Directly targets the "t2 fitted to 0.001" symptom behind W3.
2. **Kernel SEE: cheapest-guard value + attacker-origin safety** (~40 min; fixes W4 properly and sharpens W3):
   - Scalars `Vg_own/Vg_en` (cumulative product over the guard types in priority order).
   - `seeK_en/seeK_own = max_k kA_k*(g0*WIN + (1-g0)*(Vg - V_k*hitB(origin_k)))`, split by a2.
   - Prior: the enemy-Kernel to-value becomes `Vg_en`, minus a rank-1 from×to "attacker origin is hit" penalty. This replaces the constant `ek_guard`.
   - Lone-guard 2-action loss: `ownK_g1_t2 * Vg_own` (value-scaled version of v4's feature).
3. **SEE-1 for ordinary pieces with the shot/move split** (~45 min; W2 generalisation):
   - Per-type attack maps, then `shotA`, `cheapA`, and `see1 = max(Vb*shotA, Vb*anyMove - cheapA*defB, 0)`.
   - `ReduceMax` and second-max (sorting network over 5 per-type scalars), each × a2 / × (1-a2), for both sides.
   - Plus `en_att_safe_<t>` (attacker not itself hit; the SF ThreatBySafePawn analog).
   - Extending `cheapA` to v4's `hit2` reach gives the 2-exchange version.
4. **Squared Kernel-danger term with guards inside** (~45 min incl. autograd fit):
   - `-c*relu(a·[n_strike1, n_strike2, shooter_strikes, weak_esc, discovered_blockers] - b*g - b2*Vg + d)²` per side.
   - Gate by `1 - bare_K` and `shooters_alive`; fit jointly (Ethereal-style).
   - Addresses the multi-strike structure that the linear `t2_*` cannot express.
5. **Phase/a2 interactions for a small subset** (~15 min): material, the SEE maxes and Kernel danger × `ph` (nlock/round), with ridge toward 0 for the interaction columns.
6. **Data: store root search value and add off-path tree leaves; λ-blend** (~20 min code + generation time). Best combined with the `prior` direction's 800-sim runs. The gain is uncertain but compounding.
7. **Lower priority / partly redundant with v4:** per-type distance-to-Kernel tropism buckets, discovered-strike blocker counts, non-Kernel safe mobility.

**Out of scope (> 1 h or too many parameters for ~2k games):**
- Bonanza-style KPP/KKP king-relative tables.
- MMTO/comparison training of the value.
- Full multi-ply swap-list SEE with x-rays.
- A learned sigmoid lookup table for king safety (the quadratic in item 4 is the cheap substitute).

### Gaps
- Expected Elo/win-rate per item is not quantifiable from the sources. The ranking is an engineering judgement from the live-loss evidence (W2/W3/W4) and the v4 A/B results (prior-only changes ≈ 52%, value refit with 2-action terms ≈ 55%).
- None of these features has been arena-tested in this game yet.
