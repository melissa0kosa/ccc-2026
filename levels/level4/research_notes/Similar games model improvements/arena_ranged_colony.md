# Shrinking-arena, ranged-attack and Colony: One research: what transfers to our evaluator

Scope: transferable evaluation and policy ideas for firewall timing, the bare-Kernel endgame and ranged shooters, plus what is known about the real "Colony: One" game. Every idea is meant to fit in about one hour, as fixed-conv/elementwise ONNX features or as training-data changes. Written 2026-10-23, around 14:50 UTC.

Local sources used:
- `levels/level4/data/colony-one-env/docs/rules.md` (read only, as instructed)
- `levels/level4/reviews/hand_v2.md`
- `levels/level4/agents/*/RESULTS.md`
- `levels/level4/agents/hand/fast4.py` and `hand.py` (read only)
- the hyperlinks embedded in `levels/level4/level-4.pdf`

## Q1. The real game "Colony: One": official rules, strategy, and does it match ours?

### Takeaway
The real game is a small, very new Android app ("chess-like cyberpunk strategy board game") by Paul Puntschart, whose site describes him as an AI specialist. Its site publishes **no rules, no strategy and no piece values**, and I found no BoardGameGeek, Steam or community presence. There is no external strategy knowledge to mine; our `rules.md` is the only rules source.

### Cited Findings
- The contest PDF's "Colony: One – The Game" link (slide 6) points to `https://colonyonegame.wordpress.com/`. This was extracted from the PDF link annotations with pymupdf. — [level-4.pdf, page 6](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/level-4.pdf)
- The site's only description is "Colony: One is a chess-like cyberpunk strategy board game." It links to a Google Play app (`id=one.colony`). — [colonyonegame.wordpress.com](https://colonyonegame.wordpress.com/)
- The "The Game" page has no rules: only the one-line description, a developer email (colony.one.developer@gmail.com) and a post dated 17 Sep 2026. — [The Game page](https://colonyonegame.wordpress.com/the-game/)
- The creator is Paul Puntschart, who describes himself as "an artist" and is listed as an "Artificial Intelligence Specialist". He says he made the game "to play against the machine, and to play with my friends". The page gives no details on the in-game AI. — [About page](https://colonyonegame.wordpress.com/about/)
- The Google Play page could not be read: the fetch was truncated twice. Description, rating and install count are unknown. — [Google Play listing](https://play.google.com/store/apps/details?id=one.colony)
- Web searches for "Colony: One" with the piece names, and with the creator's name, returned only unrelated "Colony" titles (Bézier Games' Colony, 1987/1988 video games). — [search result example](https://whatsericplaying.com/2016/12/11/colony/), [Colony (video game)](https://en.wikipedia.org/wiki/Colony_(video_game))
- **Our rules, the only verified source:**
  - The Level 3 scene has K + Ping + Exploit per side; Level 4 has all six pieces.
  - No draws, no passing, no repetition rule.
  - The firewall locks at the end of rounds 8, 16, 24 and 26.
  - Bare-Kernel resolution happens "at the end of the current turn". Within Chebyshev 2, "the team that just finished its turn loses". Otherwise the smaller ring wins, and equal rings go to Magenta.
  - Magenta wins every tiebreak, and both Kernels burning is a Magenta win.
  — [rules.md](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/docs/rules.md)

### Inferences
- The real game is a September 2026 release by an individual developer, so community opening theory or piece values almost certainly do not exist yet. Don't spend more time on it. Our own Texel-fitted weights (hand agent) are the best "piece values" available.
- Two implications follow directly from `rules.md`:
  1. **Every game that reaches the round-26 lock is a Magenta win** (both Kernels burn). Cyan is on a hard clock.
  2. Magenta always moves last before every lock (the lock comes after Magenta's turn). This is a structural reason for Cyan's ~38% global win rate and for the value of the Cyan-specific firewall terms in hand_v3.

### Gaps
- It is not confirmed whether the app's rules (board size, firewall schedule, bodyguard order) match the contest engine. The app store text was unreadable and there is no rules page. The scene differences between Level 3 and Level 4 suggest the contest uses an adapted rule set.
- No strategy guides, forums or piece-value tables were found.

## Q2. Pommerman: what the winners did about bombs, flames and danger maps, and how they mixed search with learning

### Takeaway
The NeurIPS-2018 winners (1st and 3rd place, IBM's hakozakijunctions and dypm) used a shallow tree search (depth 1). Each leaf was extended by a **deterministic "pessimistic scenario"**: opponents are copied into every square they could reach, and the scenario runs for at least the bomb fuse length. The leaf is then scored by **counting the time-position pairs from which our agent can still survive to the end of the scenario**. This "reachable-safe-set under worst-case enemy spread" maps directly onto our fixed-conv reach maps.

### Cited Findings
- The method: "tree search where a deterministic and pessimistic scenario is used after a specified depth". Agents using it placed 1st and 3rd at the Pommerman competition at NeurIPS 2018. — [Osogami & Takahashi, arXiv 1902.10870](https://arxiv.org/abs/1902.10870)
- **How the pessimism works:**
  - It allows "objects to be located at multiple positions even if that is unrealistic or illegal"; "the opponent is copied into multiple positions in the next step".
  - Search depth is L = 1, and the scenario length is at least 10 steps (the bomb lifetime).
  - dypm counts "the number of the time-position pairs from which an agent can survive at least until the end of the scenario".
  - hakozaki scores the end positions by when the opponents could first occupy them.
  — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/1902.10870)
- **How the leaf scores are aggregated:**
  - hakozaki uses the **minimum** for own and teammate survivability and the **average** for the opponents. The paper says the average worked better than the minimum or the maximum.
  - dypm clips its own survivability at a tuned threshold to play more aggressively.
  — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/1902.10870)
- **Results:**
  - Over 1,000-match experiments per pair, the top three tree-search teams won 87.1% of games against the two learning teams.
  - Raising the pessimism level from 0 to 3 lifted dypm's win rate against dypm at level 0 from 36.9% to 77.8%.
  - Going from level 3 to level 4 raised the tie rate (17.1% to 22.7%) for only a small cut in losses: too much pessimism makes the agent passive.
  — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/1902.10870)
- Bombs explode 10 steps after placement, and flames last 2 steps. The 2nd-place learning agent (Skynet) used an "automatic reasoning module for action pruning" (danger-based pruning) on top of RL. — [Skynet paper, arXiv 1905.01360](https://arxiv.org/pdf/1905.01360)
- A comparison found that more aggressive bombing strategies were riskier "due to inadvertent suicides". — [search summary of arXiv 1905.01360 / 1911.04947](https://web3.arxiv.org/pdf/1911.04947v1)

### Inferences
- **Direct analogue for us:**
  - The "pessimistic copies" correspond to our enemy 2-action reach map `hit2` (fast4.py), which already spreads enemy pieces over every square they can reach in two actions.
  - "Survive until end of scenario" corresponds to "not on the next-lock ring at the lock, and not in `hit2`".
  - So the transferable feature is a **count of the squares our Kernel (and each piece) can reach within its remaining actions that are safe from `hit2` and not on N1 when a lock is imminent**. See Q7, idea #3.
- **Aggregation lesson:** worst case for ourselves, average for the opponent. In feature terms, use hard gates such as `[safe_count == 0]` for our Kernel, but softer, count-based terms for the enemy Kernel.
- **Pessimism-level lesson:** do not make the danger term so strong that the Kernel freezes. hand_v4b ("prior x1.5") and v4a showed no gain [hand RESULTS](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/hand/RESULTS.md), consistent with "level 4 adds ties/passivity". Scale new danger terms with the firewall urgency rather than applying them always.

### Gaps
- I did not retrieve the NeurIPS-2018/2019 learning-track winner write-ups (Navocado, 2019 winners). I can't cite their danger-map internals.

## Q3. Battlesnake royale, Hungry Geese, Lux AI and battle-royale AIs: distance-to-safe-zone vs time-to-shrink

### Takeaway
Hazard-zone bots mostly rely on **flood-fill reachability and space counts**: the squares reachable after a move, and which side reaches each square first. They tune the weights per phase with automated optimisation. I found no write-up of an explicit hazard-deadline formula. "Reachability before the deadline" is best built from the Pommerman survivability idea above combined with flood-fill space.

### Cited Findings
- **Official Battlesnake guide:** flood fill is used to find enclosed areas a snake could get trapped in. It counts the squares reachable after a candidate move. — [Battlesnake docs: useful algorithms](https://docs.battlesnake.com/guides/useful-algorithms)
- Royale is a Battlesnake variant in which hazards move in from the walls. One hobby agent combined flood-fill space with A* pathing. — [successfulsoftware.net](https://successfulsoftware.net/2021/10/13/battlesnake/)
- **snork (Leibniz University Hannover):**
  - It has "a very fast area control agent", a minimax "Tree" agent combining multiple heuristics (it placed well in royale) and a "Flood" agent. The Flood agent placed 2nd in the Elite Division of the Winter Classic Invitational 2021.
  - Heuristic influence "over time can be specified on startup", i.e. phase-dependent weights such as `size_adv_decay`.
  - Weights come from Bayesian-optimisation campaigns. The default Flood weights: space 0.0026, space_adv 0.108, size_adv 7.049, size_adv_decay 0.041, food_distance 0.173, health 0.00044.
  — [wrenger/snork](https://github.com/wrenger/snork)
- A Battlesnake paper uses iterative deepening with alpha-beta and max^n, and a "diamond flood filler" for board control. Its parameters were tuned with grid search and a genetic algorithm. — [GI paper on Battlesnake search](https://dl.gi.de/items/43c71d90-3b16-4a1c-93c2-c1a98104a283/full)
- A 2018 open-source snake uses flood fill as a last resort: it moves into the largest component when its safety heuristics fail. — [okjuan/snake2017-2018](https://github.com/okjuan/snake2017-2018)

### Inferences
- **What transfers:**
  1. A **space advantage term**, not just binary escape: the number of non-N1, non-`hit2` squares around each Kernel, and own minus enemy.
  2. **Phase-decayed weights.** snork's "influence over time" corresponds to our u/u1/u2/nlock interactions, which hand already uses.
- **The snork weights show a pattern:** a large coefficient on a coarse advantage (size_adv about 7) plus small coefficients on space terms. For us, keep material/Kernel-capture terms dominant, and make the new space terms small corrections that the Texel fit can find.
- **Deadline arithmetic specific to our engine** (my analysis from rules.md):
  - The Kernel, Ice and Exploit can leave any ring square in 1 action: Kernel or Ice diagonally inward, the Exploit by L-jump.
  - The orthogonal-only movers (Ping, Daemon, Trace) on a **corner of the outer open ring need 2 actions**, because every orthogonal destination from a corner stays on the same ring. Check: (0,0) -> (0,1..3) or (1..3,0) are all ring 0.
  - The Trace also needs empty squares.
  - So "minimum actions to safety" per piece is 0, 1, 2 or ∞ (blocked) and is computable from fixed masks. See Q7, idea #5.

### Gaps
- No primary Hungry Geese (Kaggle) or Lux AI day/night winner write-up was retrieved; the search returned no relevant pages. I cannot cite how they valued "time to night/hunger" against distance.
- Battle-royale zone AIs for shooters (PUBG-like) were not researched for lack of time.

## Q4. Ranged-attack board games: evaluating line-of-fire threats, blocking, and pieces pinned against a shooter

### Takeaway
I found no engine documentation for Khet, Rifle chess or Fairy-Stockfish shooting variants. The decisive transferable point comes from our own rules, and our code currently misses it. **A shot (Trace at 2-3 diagonal, Daemon adjacent) captures without moving, so defending the target does not deter it.** Our "hanging" features and the prior's danger term treat any attacked-and-defended piece as safe.

### Cited Findings
- **Our rules:**
  - The Trace "shoots diagonally at range 2-3"; "an adjacent diagonal piece blocks the line and cannot be shot; only the first piece at range 2-3 is a candidate."
  - The Daemon "shoots adjacent enemy" and move-captures at 2-3.
  - An attack on a Kernel sacrifices a guard, and "the attacker remains at its origin even if the action was a move-capture."
  — [rules.md](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/docs/rules.md)
- **Our code:**
  - fast4.py defines `D = hit_own * hit_en` (attacked and defended).
  - The hanging features are `Phang = own5 * (hit_en - D)`.
  - The prior uses `danger = (hit_en - D)*P["danger"] + D*P["danger_def"] + ...`.
  - So a piece that is **defended and attacked only by a shooter** counts as "not hanging".
  - This holds whichever piece attacks: `hit_en` merges all attack types.
  — [fast4.py lines 209-220, 276](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/hand/fast4.py)
- **Live evidence that shots decide games:**
  - 3 of our 10 live wins were Kernel captures by a Trace shot.
  - Loss 6032 a37-a38: C3->C4, then the shot C4xF7 killed our last guard, which led to the bare-Kernel loss.
  — [hand_v2 review](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/reviews/hand_v2.md)
- **Game of the Amazons (the classic "shoot without moving into the square" game):** engines evaluate territory as the squares one side reaches first by distance, plus mobility, with weights that vary by game phase. — [chessprogramming: Amazons](https://chessprogramming.org/Amazons); [arXiv 2603.10512](https://arxiv.org/html/2603.10512v1)

### Inferences
- **Shot-aware hanging (the highest-value finding; see Q7, idea #1):**
  - Split `hit_en` into `hit_en_shot` (Trace diagonal 2-3 first-piece map, Daemon adjacent map) and `hit_en_move`.
  - True hang = `clamp(hit_en_shot + (hit_en_move - D_move), 0, 1)`.
  - In chess SEE terms, a shot is an exchange with no recapture on the target square. The only recourse is to attack the shooter itself, which is a separate threat.
- **The Trace "screen/x-ray" geometry is Xiangqi-cannon-like, inverted:**
  - A piece diagonally adjacent to an enemy Trace is **immune** to it and also shields the whole diagonal behind it.
  - With 2 actions per turn, "capture the screen, then shoot through" is a 2-action strike. It is not caught by `hit2` (move then attack), because the line opens only after the capture.
  - The feature `xray_T(target) = Σ_dirs enT(s) * occ(s+d) * own_valuable(s+2d or s+3d)` costs 4 diagonal shift kernels.
- **For Kernel attacks, every attacker behaves like a shooter** (it stays at its origin). One available attack plus ≤1 guard therefore means the Kernel dies this turn (two attacks). hand already encodes this as `thr_own = g0*... + g1*ka`. Under W4, the prior bonus for a Kernel attack should still scale with the number of guards (already done in v4: `ek_guard`).

### Gaps
- No primary engine sources were found for Khet/laser chess, Rifle chess, or the Fairy-Stockfish variant list's shooting pieces. I can't cite how those engines score line-of-fire or pins.
- The Amazons queen-distance/king-distance papers (Hashimoto 2001, Lieberum 2003) were not retrieved.

## Q5. "Closer to the centre wins" endgame: race evaluation, tempo/parity, and whether an exact bare-Kernel solver fits in as a feature

### Takeaway
No race or tempo counting is needed. Per `rules.md`, the bare-Kernel position is resolved **once, statically, at the end of the turn in which the last non-Kernel piece disappears**. The exact outcome is a closed-form function of the two Kernel rings, their Chebyshev distance, who finished the turn, and the colour. A two-line elementwise feature computes it exactly, so no table is needed. The leverage is in the **"one capture away from bare"** positions, i.e. the trade-down decision.

### Cited Findings
- **The resolution rule:**
  - "When only both Kernels remain, resolve at the end of the current turn."
  - Within Chebyshev distance 2, "the team that just finished its turn loses".
  - Otherwise the smaller ring (closer to the centre) wins. Equal rings: Magenta.
  — [rules.md](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/docs/rules.md)
- hand already has `Kdepth_diff_bare = (dk_own - dk_en) * relu(1 - S(ownNK) - S(enNK))`, with weight 1.5 in hand.py. It **only fires when no non-Kernel piece is left at all**. — [hand.py line 266, 368](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/hand/hand.py)
- **Bare endings happen:**
  - In hand_v2 vs stopgap_v1 (80 games), 8 wins were by bare Kernel.
  - Live losses 6509 and 6032 both went through "bare Kernel vs K+X" phases.
  — [hand RESULTS](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/hand/RESULTS.md); [hand_v2 review](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/reviews/hand_v2.md)

### Inferences
**Exact "resolution if I make the last capture now" feature (fixed ops):**
- `dr = S(ownK*ring_idx) - S(enK*ring_idx)`. This is an integer, and nlock cancels.
- `ring_sign = clamp(2*dr + (2*mag - 1), -1, 1)`. This gives +1 if we are deeper, -1 if we are shallower, and on equal rings +1 for Magenta and -1 for Cyan.
- `ched = S(ownK * clamp(conv(enK, ones5x5, pad=2), 0, 1))`. This is the same 5x5 kernel as W2's `kz2_en`.
- `res_cap_own = (1 - ched)*ring_sign - ched`. This is our result if **we** finish the turn in the bare position: within 2, we, the finisher, lose.
- `res_cap_en = (1 - ched)*(-ring_sign) + ched`. This is our result if **they** finish it: within 2, they lose.

**Gates:**
- `lastE = [nE == 1]` and `lastO = [nO == 1]`, as `relu(1 - |S(enNK) - 1|)`.
- `capE = clamp(S(enNK * hit_own), 0, 1)`: their last piece is attackable by us.
- `capO = clamp(S(ownNK * hit_en), 0, 1)`: our last piece is attackable by them.

**Value features:**
- `bareE = lastE * [nO <= 1] * capE * res_cap_own`
- `bareO = lastO * [nE <= 1] * capO * res_cap_en`

Texel-fit them, or start at about 0.8.

**Why it matters:**
- In K+1 vs K+1, capturing their last piece **ends the game at the end of our turn**. If we are on the worse ring, the capture is a loss.
- Magenta can trade down whenever the rings are equal and the Kernels are more than 2 apart.
- In bare-K vs K+X, the bare Kernel's only counter-chance is to capture X (it then finishes the turn, so it needs distance > 2 and a deeper ring). The side with K+X should keep X out of reach of the enemy Kernel and keep its own Kernel deep.

**Prior term:** `to += w_res * lastE_map(to) * res_cap_own` (w about 2), where `lastE_map = enNK * lastE`. This vetoes "winning-looking" last captures that lose the resolution, and promotes the winning ones.

**Approximations:**
- The capture itself can change the ring of our Kernel (if the Kernel captures).
- With `a2` the finisher still has one action to step the Kernel away or inward.
- Both are tolerable for a Texel feature. For exactness, compute `ched`/`dr` on the post-move Kernel map, which only applies to Kernel captures.

## Q6. Phase/timing features: rounds-to-lock × distance-to-safety, and how value weights should change as the lock approaches

### Takeaway
Use **"actions-to-safety vs actions-before-lock"** per piece. For the Kernel, add a **pessimistic safe-escape count** gated by u2/u1, rather than a flat u2 bonus. Magenta moving last before each lock means Cyan's effective deadline is one opponent turn earlier: Magenta can attack or body-block Cyan's escape squares after Cyan's last move. The live games mostly end around the first lock (mean 33 actions, about round 8), so the **round-8 lock is the one that matters**.

### Cited Findings
- The lock happens "at the end of rounds 8, 16, 24, and 26"; a round is Cyan's turn and then Magenta's; locked squares are permanent, and pieces on them are destroyed. — [rules.md](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/docs/rules.md)
- **Existing timing terms in fast4.py:**
  - `u1 = relu(u - 0.875)*8` is "shrink this round"; `u2 = clamp((u - 0.625)*8, 0, 1)` is "within 2 rounds".
  - The value already has okN1/ovN1/evN1 × {u, u1, u2, colour} and `noesc_own_u1`.
  - The prior has `fw_from`/`fw_to` gated on u1 only, plus `k_fw2 * ownK * N1 * u2` (the v4 W5 fix).
  — [fast4.py lines 36-63, 233-275](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/hand/fast4.py)
- **Live evidence:**
  - Round-8 burns fell to 1 piece in 13 games with hand_v2, from 4 pieces in 3/5 Cyan games with stopgap.
  - The remaining firewall gap is the Kernel staying on the corner ring in rounds 6-7: 6585 a26 A1->A2 at Q +0.65 vs +0.33; 6754 a22 at +0.68 vs +0.47.
  - Mean game length is 33 actions.
  — [hand_v2 review](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/reviews/hand_v2.md)
- **The v4 bundle results:**
  - The v4 bundle (incl. k_fw2 0.6) with a refit: 55% vs v3 [48,61].
  - Without a refit: 38%.
  - So prior-only bonuses without a value refit hurt.
  — [hand RESULTS](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/hand/RESULTS.md)
- **Phase-dependent weights are standard in zone games:**
  - snork specifies each heuristic's "influence over time".
  - Amazons evaluators vary territory/position/mobility weights by phase.
  — [snork](https://github.com/wrenger/snork); [arXiv 2603.10512 summary](https://arxiv.org/html/2603.10512v1)

### Inferences
**Deadline arithmetic:**
- Let `A_left` be own actions remaining before the next lock. It is roughly `2*rounds_to_lock + (a2 ? 1 : 2)` for the side to move, derived from u, a2 and colour.
- `d_p` is the minimum actions to leave N1:
  - Kernel, Ice, Exploit: 1 (if a free inner square exists).
  - Ping, Daemon, Trace: 1 on an edge, **2 on a corner** of the open ring.
  - ∞ if every route is blocked.
- `debt = Σ_p own_p*N1*d_p`. Value term `relu(debt - A_left)` × cheapest-at-risk value (it is 0 except in the last 1-2 turns). Prior: "must-move-now" from-bonus for pieces with `d_p >= A_left`.

**Pessimistic Kernel escape (from Pommerman):**
- `safeK = (1 - hit2_en) * (1 - N1*u2) * empty_or_capturable * (1 - locked)`.
- `K1 = clamp(conv3x3(ownK) * safeK)`, `K2 = clamp(conv3x3(K1) * safeK)`.
- Features: `ownK_noSafe_u2 = [S(K2) == 0] * u2` and `ownK_safe_cnt = log1p(S(K2))`, plus the enemy mirrors with averaged (soft) weights.
- Prior to-term for Kernel moves: `+w * conv3x3(safeK)(to) * u2`.

**Doomed material:**
- In the lock round (u1), a piece on N1 that cannot escape before the lock is worth about 0.
- Capturing it is a wasted action. A **move-capture onto N1 in the last turn before the lock kills the capturer**; a shot does not.
- Gate `ovN1_u1`/`evN1_u1` by `(1 - can_escape)` (fixed), and penalise move-captures onto N1 under u1 in the prior. The existing `fw_to` partly does this; leave shots ungated.

**Cyan clock:**
- Reaching the round-26 lock = Magenta wins.
- This has low priority because games average about 8 rounds, but `cy * [nlock == 3]` is a free feature for completeness.

## Q7. What transfers to our evaluator, ranked by expected gain per hour

### Takeaway
Cheapest high-value items:
1. Shot-aware hanging/danger: a correctness gap in our exact maps.
2. Oversampling firewall-critical and near-bare positions in the existing fitting and distillation pipelines.
3. A Pommerman-style pessimistic Kernel safe-escape count under u2.
4. The exact pre-bare resolution feature.

Everything else is marginal or out of scope.

### Cited Findings
- Arena noise: 40-60-game arenas carry ±12-15 points of noise. hand A/Bs use 200 games (CI about ±7). — [az RESULTS](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/az/RESULTS.md); [hand RESULTS](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/hand/RESULTS.md)
- Prior-only changes without a value refit lost (v4p 38%). The refit v4f gained 55%. — [hand RESULTS](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/hand/RESULTS.md)
- The ONNX budget is a cap of 1024 nodes and 512 initializers. FastNet4 uses 238/286, so there is room for about 10 more convs. — [hand RESULTS](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/hand/RESULTS.md); [hand_v2 review](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/reviews/hand_v2.md)

### Inferences
Ranked list (my estimates, not measured):

1. **Shot-aware hanging + danger (value + prior), about 45 min.**
   - Build `hit_en_shot` from the per-type attack maps: Trace diagonal first-piece-at-2-3 with the adjacent blocker, and Daemon adjacent.
   - Value: `own_hangS_<X> = S(own_X * hit_en_shot * hit_own)`, i.e. the newly counted "defended but shot" cases, and the enemy mirror. Texel refit with `fit4.py`.
   - Prior: `danger += hit_en_shot * hit_own * (P["danger"] - P["danger_def"])`.
   - **Test:**
     - (a) Count how often `own_X * hit_en_shot * hit_own > 0` occurs in v1-v3 self-play. If it is under 1% of positions, deprioritise.
     - (b) A hand-built position with a defended Ice on an enemy Trace diagonal: the old feature says 0, the new one says 1.
     - (c) A 200-game A/B against hand_v3/v4f.
2. **Training-data weighting, about 15 min, zero graph risk.**
   - In Texel (`fit4.py`) and in the prior agent's 800-sim distillation, upweight these positions x2-3:
     - (i) u2 > 0 with any own piece on N1 (especially the Kernel);
     - (ii) total non-Kernel pieces ≤ 2;
     - (iii) any piece under `hit_en_shot`.
   - In prior/az data generation, add **seeded starts**: self-play from positions at rounds 6-7 with the Kernel on the outer ring or a corner. This is the W5 pattern; the 800-sim labels then teach the early step-off. The value refit makes prior changes stick (v4p vs v4f).
3. **Pessimistic Kernel safe-escape count (Pommerman dypm analogue), about 30 min.**
   - `K2` safe reach as in Q6, built on the existing `hit2` maps, gated by u2. Features `ownK_noSafe_u2` / `ownK_safe_cnt` plus mirrors. Prior Kernel to-term `+w*conv3x3(safeK)*u2` in place of (or alongside) the flat `k_fw2`.
   - **Test:**
     - Positions 6585 a26 and 6754 a22: A1->A2 should become prior top-1, or at least > 0.2.
     - 200-game A/B.
     - Keep the weights moderate. Pommerman shows that excess pessimism makes play passive.
4. **Exact pre-bare resolution features `bareE`/`bareO` plus the prior last-capture veto, about 30 min.**
   - Formulas are in Q5; they reuse W2's 5x5 `kz2_en` kernel and `ring_idx`.
   - **Test:**
     - Unit positions: K+T vs K+P with our Kernel on the shallower ring. The prior on capturing the last piece should drop, and the value should be < 0.
     - Same capture with our Kernel within Chebyshev 2 of theirs: the capture should be vetoed.
     - Then A/B. Expected gain is limited to the ~10% of games that reach bare or near-bare phases.
5. **Evacuation debt for orthogonal movers on ring corners, about 30 min.**
   - Fixed `CORNER` mask × N1 × (P, D, T planes) → `d_p = 2`.
   - `debt` vs `A_left` as in Q6. Prior "must-move-now" bonus for corner orthogonal movers when `A_left <= 2`.
   - Expected gain is small: live burns are already down to 1 piece per 13 games.
6. **Trace x-ray/screen feature, about 45-60 min, borderline.**
   - 4 diagonal-shift kernels: an enemy Trace with an occupied adjacent diagonal square and our valuable piece (or Kernel guard) at distance 2-3 behind it.
   - Value-only, Texel-fitted. It catches the "capture screen then shoot" 2-action strike that `hit2` misses. Do it only if #1 shows that shot tactics are frequent.
7. **Doomed-material gating (u1 × no-escape) and the Cyan round-26 clock, about 15 min, low gain.** Both are mostly covered by the existing evN1/ovN1 × u1 features, and games rarely reach round 16+.

**Out of scope (> 1 hour or not ONNX-fixed):**
- Full pessimistic-scenario rollouts inside the evaluator: the referee search is fixed, so this would need multi-step simulation in the graph.
- Learned danger subnetworks trained from scratch.
- Any tablebase beyond the closed-form bare rule. None is needed: the rule is a static function at the end of the turn.

### Gaps
- None of the ranked gains have been measured. The frequency check in #1(a) is the cheapest way to tell whether the top item matters before spending 45 minutes on it.
- Whether Daemon's "adjacent" shot includes diagonals is not explicit in rules.md. Check the engine's attack-map code in `hand.py` (which already matches the engine) before building `hit_en_shot`.
