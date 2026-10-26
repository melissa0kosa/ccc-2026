# Evaluating, gating and deploying models in a live Glicko-2 AI-vs-AI tournament (CCC 2026 L4, COLONY: ONE)

Scope: how to tell quickly and reliably whether a candidate is better, which model to put in the single seat, and how Glicko-2 behaves in the 1-3 hours left. Local numbers come from our own replays and dashboard dump (`levels/level4/data/matches/`) and from a small single-core calculation: Wald SPRT plus a simulation of the Glicko-2 algorithm exactly as in Glickman's paper. Script: `C:/Users/melis/AppData/Local/Temp/claude/C--Users-melis-Downloads-cloudflight-coding-contest/2537b5b1-8fcc-4ac8-8fd3-b627f7b7b210/scratchpad/calc.py`.

## Q1. Sequential testing: how many games separate 55/60/70% from 50%, and how do we cut eval time?

### Takeaway
With 1 s CPU per game, we can afford to detect edges of about 65-70% or more (roughly 20-60 games) plus a gauntlet. A 55% edge needs about 350-530 games on average, so don't try to prove it in the time left. Run the SPRT on game pairs (same opening seed, colours swapped), and count distinct games rather than nominal games, because the referee is near-deterministic.

### Cited Findings
- SPRT stopping thresholds are a ≈ log(β/(1−α)) and b ≈ log((1−β)/α). They are exact only for continuous sampling, because the LLR "may cross the threshold between samples" — [Wikipedia: SPRT](https://en.wikipedia.org/wiki/Sequential_probability_ratio_test). With α = β = 0.05 this gives LLR bounds of [−2.94, +2.94]; with α = β = 0.10 it gives [−2.20, +2.20]; with α = 0.05, β = 0.20 it gives [−1.56, +2.77] (own calculation from that formula).
- Stockfish development treats SPRT as "generally regarded as the most important process in modern engine development". The bounds listed there are Elo [0, 2] and [0, 3] (normalized Elo) for its LTC/STC tests. fast-chess "supports pentanomial statistics, which can make tests finish faster". Balanced opening books are recommended for weaker engines and biased books for strong ones (to cut draws) — [Chessprogramming wiki: SPRT](https://www.chessprogramming.org/Sequential_Probability_Ratio_Test).
- Fishtest uses a Generalized SPRT, with MLEs plugged in for nuisance parameters. Its pentanomial model counts game pairs (LL, LD, DD/WL, WD, WW) and is "a more accurate representation than the traditional trinomial model", giving "a substantial saving of testing resources". Normalized Elo makes the "expected duration of a test dependent only on the chosen bounds" — [Fishtest Mathematics](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html).
- **Expected games (Bernoulli SPRT, H0 p = 0.50 vs H1 p = p1, no draws; Wald ASN, checked with 3000-run simulations)** — own calculation:

  | α = β | LLR bounds | p1 = 0.55 | p1 = 0.60 | p1 = 0.70 |
  |---|---|---|---|---|
  | 0.05 | ±2.94 | mean 530, median 430, 90th pct 1040 | mean 132, median 108, 90th pct 261 | mean 32, median 27, 90th pct 65 |
  | 0.10 | ±2.20 | mean 351 (90th pct 700) | mean 87 (90th pct 186) | mean 21 (90th pct 45) |
  | α = 0.05, β = 0.20 | −1.56 / +2.77 | 381 if H1 true, 267 if H0 true | 95 if H1, 66 if H0 | 23 if H1, 15 if H0 |

  The test runs longest when the true p lies between p0 and p1, so use p1 = 0.65-0.70 and accept that a real 55% edge will often end as "reject" or hit the game cap.
- **Fixed-n Wilson 95% gate (lower bound > 0.5)** — own calculation: you need ≥ 27/40 (67.5%), ≥ 38/60 (63%) or ≥ 60/100 wins. For 80% power, n ≈ 194 games at true p = 0.60, about 47 at 0.70 and about 13 at 0.85.
- Applied to our current results — [hand RESULTS.md](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/hand/RESULTS.md), [az RESULTS.md](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/az/RESULTS.md):
  - hand_v2 vs stopgap_v1: 68/80, Wilson lower bound 0.756, a clear pass.
  - hand_v1 vs stopgap_v1: 31/60, lower bound 0.393, effectively equal.
  - az_v1 vs stopgap_v1: 6/60, a clear reject.
- The referee plays deterministically (greedy by visits) after 8 opening actions that use noise and temperature, so two fixed models produce few distinct games — [STATEMENT.md](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/STATEMENT.md).

### Inferences
- **Effective n = number of distinct games.** Hash each game's action list. If 60 games contain only 15 distinct lines, the true Wilson interval is about √(60/15) = 2× wider than reported. Fix this by widening opening randomness in the local arena: use more noisy plies, or start from a small book of 20-40 random 2-4-ply openings. Give each opening to both colours (a "pair", as with `-repeat` in fast-chess) so the colour advantage cancels exactly.
- **Pairs alone do not shrink variance much here.** With independent games and a 38/62 colour split, the variance per pair is 0.118 against 0.125 for two unpaired games, about a 6% saving (own calculation). The real gain comes from correlation within a pair, because a near-deterministic engine makes the opening decide much of the result. Then pentanomial or pair-level statistics (WW / split / LL) finish faster, as Fishtest reports.
- **Common random numbers.** When comparing candidate A against incumbent B, have both play the same gauntlet opponent on the same fixed list of opening seeds. Compare per seed (a McNemar-style count of discordant seeds) instead of comparing two independent win rates.
- **Information per game peaks at p ≈ 0.5**, because Fisher information is proportional to p(1−p). A game at p = 0.95 carries about 1/5 of the information of a 50% game. Sparring partners far below the candidate saturate: az_v1 scored 90% against seat_l3 but 10% against stopgap. Spend most games on opponents within ±150 Elo.
- Practical gate given the CPU budget, one core at about 1 s per game:
  1. Run an SPRT on pairs, H0 = 0.50 vs H1 = 0.65, α = β = 0.10. Expect about 40-60 games (~1 min), capped at 120 games.
  2. Then run a 20-game regression check against each gauntlet member.

### Gaps
- Lc0's own testing methodology page was not fetched. It probably uses the same cutechess/SPRT tooling, but this is unverified.
- No source gives game counts for pentanomial SPRTs as a function of opening correlation; the saving has to be measured on our engine. Log WW/split/LL counts.

## Q2. Non-transitivity and overfitting to one sparring partner: what minimal gauntlet?

### Takeaway
"Beats the previous best" is necessary but not sufficient. Real games have a transitive axis plus cycles at mid strength, which is where most of the field sits, so gate on a small, diverse gauntlet near our strength and reject any candidate with a large regression against any member.

### Cited Findings
- Czarnecki et al. find that real games have a "spinning top" geometry: transitive strength on the vertical axis, and many non-transitive cycles at intermediate strength on the radial axis. This geometry explains "why populations of strategies are necessary for training of agents" — [Real World Games Look Like Spinning Tops (NeurIPS 2020)](https://arxiv.org/abs/2004.09468). A follow-up on more than a billion Lichess/FICS games finds the same geometry in human chess — [Measuring the Non-Transitivity in Chess](https://ar5iv.labs.arxiv.org/html/2110.11737).
- AlphaStar's league has three roles. Main agents train against themselves and all league members, for robustness. Main exploiters target the current main agent's weaknesses. League exploiters look for weaknesses of the whole league. The authors' point is that "playing to win is insufficient" — [DeepMind AlphaStar blog](https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/). PFSP weights opponents by f(win-rate) = (1−x)^z with z = 6, as reported by a reimplementation citing Vinyals et al. 2019 — [TStarBot-X](https://arxiv.org/abs/2011.13729) (secondary source).
- Balduzzi et al. propose Nash averaging, a maximum-entropy Nash meta-game over the evaluation matrix that is "invariant to redundant tasks and agents". It allows maximally inclusive evaluation, because duplicates of weak agents no longer skew the result. They also propose mElo for cyclic interactions — [Re-evaluating Evaluation (NeurIPS 2018)](https://arxiv.org/abs/1806.02643).
- Our own evidence: hand_v1 scored 95% against seat_l3 but 52% against stopgap, and az_v1 scored 90% against seat_l3 but 10% against stopgap. Against a weak partner, win rates saturate and stop separating models — [hand RESULTS.md](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/hand/RESULTS.md), [az RESULTS.md](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/az/RESULTS.md).
- The environment ships only `play/random.onnx` as a reference model. There is no default AlphaZero baseline, so "default" opponents are whatever each team trained (`levels/level4/data/colony-one-env/play/`, directory listing).

### Inferences
- Minimal gauntlet. All entries already exist in `levels/level4/models/` except where noted.
  1. random.onnx and greedy, as a sanity floor. Require ≥ 95%; 10 games each is enough.
  2. seat_l3_material, the old live seat and a proxy for a weak but plausible field. Require ≥ 85%.
  3. stopgap_v1, the deployed incumbent. Use the SPRT gate from Q1.
  4. hand_v1, a different hand-designed style.
  5. az_v1 and az_v2, learned AlphaZero-style play. These are the most likely stand-in for other teams.
  6. A deliberately different style: a "kernel rusher" (maximise immediate kernel-capture threats) and/or a "turtle" (keep pieces adjacent to the kernel, play for the Magenta tiebreak). This takes about 15 min to build by reweighting the hand evaluator's terms.
- Gate rule:
  - Pass the SPRT against the incumbent.
  - Do not lose more than about 10 percentage points against any gauntlet member versus the incumbent's score there (20 paired games each).
  - Score ≥ 95% against the sanity bots.
- A full Nash-averaging matrix is overkill for a 1-3 hour horizon. Even a 6×6 cross-table of 20 games per cell is about 15 CPU-min, and the "no big regression" rule captures most of its value. If time allows, compute the max-entropy Nash over the cross-table and check that the candidate carries most of the Nash mass.

### Gaps
- No direct evidence that COLONY: ONE has strong strategy cycles. Our data so far fits a transitive chain: l3 < az_v1 < stopgap ≈ hand_v1 < hand_v2.
- Kaggle simulation-competition writeups on leaderboard dynamics were not found in reliable or primary form. Only an unofficial TrueSkill blog post and the Ladder paper on leaderboard overfitting turned up ([Blum & Hardt, The Ladder](https://www.arxiv.org/abs/1502.04585)).

## Q3. Glicko-2 mechanics: upload effects, gain per game, games per hour, when to replace and how late

### Takeaway
The live system behaves like textbook Glicko-2 with one game per rating period. After about 16 games the RD is about 90 and each game moves the rating about ±24. Our seat plays about 100-150 games per hour, so a stronger model closes about 85% of its rating gap in about 50 games (20-30 min). Uploading a clearly better model is cheap: the rating is kept, and higher volatility only speeds convergence. The real risks are a regression and too few games before the end. Even at constant strength, the final rating carries ±45-70 points of pure luck.

### Cited Findings
- Glicko-2 basics — [Glickman, Example of the Glicko-2 system](http://www.glicko.net/glicko/glicko2.pdf):
  - Unrated players start at r = 1500, RD = 350 and σ = 0.06. τ should be between 0.3 and 1.2, with smaller τ for "extremely improbable collections of game outcomes".
  - Glicko-2 scale: μ = (r − 1500)/173.7178 and φ = RD/173.7178.
  - Before each period, φ* = √(φ² + σ′²); after it, φ′ = 1/√(1/φ*² + 1/v), and μ′ = μ + φ′² Σ g(φj)(sj − E).
  - The system "works best when the number of games in a rating period is moderate to large, say an average of at least 10-15 games per player".
  - Players who don't play only get their RD inflated.
- **Live server = one game per period.** The dashboard's rating series has one point per game: all 18 exported replays match a point by exact timestamp, and the 2 extra points (13:43:38, 13:43:43) bring the total to the leaderboard's 20 games. Our first game, a loss, took the rating from 1500 to 1305 (−195), which fits RD = 350. The implied RD fell to about 90-100 by games 14-18, with swings of ±21-29 per game — [tournament-dashboard.json / tournament-matches.json](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/matches/tournament-dashboard.json); own calculation. A simulation of standard Glicko-2 (σ = 0.06, τ = 0.5) gives RD ≈ 91 after 16 games, consistent with the server using defaults.
- **Game rate.**
  - Our seat played 18 games in 7.2 min (13:35:02 to 13:42:15), about 150 per hour.
  - The dashboard shows 20 games by 13:43:43, about 138 per hour.
  - Globally, 100.4 games/min across 127 active kernels means about 1.6 games/min per seat, about 95 per hour — [tournament-dashboard.json](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/matches/tournament-dashboard.json).
  - So the 8 games needed to show a score take only about 4-5 min.
- **Expected change per game** at rating parity, RD ≈ 92 against opponents with RD ≈ 80 (own calculation, Glicko-2 formulas): +2.4 at a 55% true win rate, +4.8 at 60%, +9.5 at 70%. A win or loss is worth ±24.
- **Converting win rate to Elo:** 55% ≈ +35, 60% ≈ +70, 70% ≈ +147, 85% ≈ +301.
- **Convergence after an upload.** 400 simulated runs, starting at our 1161 with RD 95 and opponents matched within about ±60 (own calculation):

  | Scenario | after 16 games | after 50 games | after 120 games | final SD |
  |---|---|---|---|---|
  | true 1300, σ = 0.06 | 1236 | 1280 | 1295 | about 42-51 |
  | true 1300, σ = 0.15 (volatility raised) | 1253 | 1299 | — | about 63-69 |
  | true 1400, σ = 0.06 | 1287 | 1367 | 1401 | — |
  | true 1400, σ = 0.10, RD 150 | 1340 | 1391 | — | about 54-63 |
  | true = current 1161 (no real change) | — | — | — | about 42-49 |

  The last row shows the noise floor of the final rating with no change at all.
- The observation says a new upload keeps the rating, raises volatility and restarts the 8-game count; the final ranking uses the rating at contest end — [STATEMENT.md](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/STATEMENT.md). The exact raised σ and τ values are not published.

### Inferences
- **Upload hand_v2 (or whatever passes the gate) now, not later.** Our 1161 rating was earned by the old L3 seat. hand_v2 beats the deployed stopgap 85% (about +300 Elo head-to-head; less against the field), so every hour of delay leaves 100+ games of climbing on the table.
- **Late uploads.** Since the rating is kept and games are fast, a late upload is fine if at least about 50 games remain (≥ 30 min for a big edge; ≥ 45-60 min for a small one). Don't upload in the last 15 min:
  - the 8-game count may not complete, and how the final standings treat that case is unknown;
  - the raised volatility makes the last games swing more;
  - a hidden regression can't be noticed or rolled back.
- **When to replace a decent seat.** Replace only if the candidate:
  - passes the SPRT against the incumbent at H1 = 0.65; and
  - shows no gauntlet regression of more than 10 points (Q2).
  A small edge (≤ 55%, about +35 Elo) is below the ±45-70 point noise of the final rating, and raising σ adds variance. Its expected gain is small but positive, so it's worth uploading only early.
- **Monitor live:** watch the first 20-30 live games of a new upload. At ±24 per game, 10 net losses cost 240 points. If the live win rate is ≤ 35% over 20 games while the predicted rate against similar ratings is about 50%, roll back. Re-uploading the old model keeps the rating anyway.
- Matchmaking pairs similar ratings, so each upload is effectively tested against the stratum we currently sit in. As we climb toward 1800-2000 (top of the table: CVCT 2020, NicoSeverin 1943), the opponents get stronger and our win rate flattens toward 50%.

### Gaps
- The server's τ, the σ applied on upload, whether RD is also reset, and opponent RDs are unknown; the simulations assume σ0 ∈ {0.06, 0.10, 0.15}, τ = 0.5 and opponent RD = 80.
- Whether a seat with fewer than 8 games at the end still counts in the final ranking is not documented.

## Q4. Colour-specific play: should value or policy differ by colour?

### Takeaway
Yes. The tiebreak rule makes the game asymmetric, and the observation already includes a "Magenta to move" plane, so a correct value function must depend on colour. Terminal and near-terminal tiebreak states are +1 for MAGENTA and −1 for CYAN. As MAGENTA, play for simplification and the tiebreak; as CYAN, avoid trades into bare-kernel endgames and commit to winning by capture.

### Cited Findings
- The observation is canonicalised to the side to move: Magenta's board is rotated 180 degrees, and planes 0-5 always hold the mover's pieces. Plane 15 is 1.0 when Magenta is to move, "because Magenta wins rule tiebreaks". Plane 14 encodes firewall urgency. Value is from the mover's perspective, and the game has no draws — [observation.md](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/docs/observation.md).
- MAGENTA wins all tiebreaks (both kernels burnt, or a same-ring bare-kernel endgame), and the global CYAN win rate is 0.380 over 4187 games — [STATEMENT.md](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/STATEMENT.md), [tournament-dashboard.json](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/matches/tournament-dashboard.json).
- The old live seat went 0/9 as CYAN and 7/9 as MAGENTA in the 18 exported replays (own parse of the `*** X WINS ***` lines). hand_v2 against stopgap went 82% as CYAN and 88% as MAGENTA, with 8 of its 68 wins coming from bare-kernel tiebreaks — [tournament-matches.json](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/matches/tournament-matches.json), [hand RESULTS.md](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/hand/RESULTS.md).

### Inferences
- **Value function.** Add or verify colour-signed terms:
  - "both kernels will burn": +1 if plane 15 = 1, otherwise −1;
  - "bare-kernel same-ring endgame reachable";
  - a material-trade term with opposite sign by colour, with MAGENTA favouring trades.
  This works like contempt in chess engines, where draw-ish outcomes are scored asymmetrically.
- **Evaluation.** Report win rates per colour. Each pairing gives one game per colour, so a 10-point gain as CYAN (38% → 48%) is worth exactly as much rating as a 10-point gain as MAGENTA. Our old seat's 0/9 as CYAN suggests the cheapest wins are in CYAN games (avoid firewall and kernel blunders).
- **Learned models.** A single network can learn colour-dependent value from plane 15, but only if self-play reaches enough tiebreak endings. Hand evaluators need explicit terms.

### Gaps
- No reliable external source on colour-dependent policies in asymmetric-tiebreak games beyond the general chess "contempt" idea, which we did not verify with a fetched source.

## Q5. Learning from our own live replays: regression tests and quick fixes

### Takeaway
Turn every live loss into a position-level regression test: the position just before the decisive blunder, plus the property the chosen move must satisfy. Run the suite (seconds) before every upload. The round-8 firewall kernel lock is the single largest systematic loss source and should be a hard gate.

### Cited Findings
- In the 18 exported replays, our kernel is listed among the pieces burnt at FIREWALL ROUND 8 in 5 games by a strict own parse; the team's analysis counts 8 of 18 firewall-at-r8 losses. The other losses were early captures of an unguarded kernel by Ping, Ice, Exploit or Daemon — [tournament-matches.json](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/matches/tournament-matches.json), [STATEMENT.md](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/STATEMENT.md).
- hand_v2's 12 losses against stopgap over 80 games break down as 11 captures and 1 round-8 firewall loss. Its features already include "firewall next-ring" and "firewall escape" terms — [hand RESULTS.md](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/hand/RESULTS.md).
- Replays include `starting_observation` and a full `protocol` move list (e.g. `TR B4 -> E7 SHOOT xEX`, `---- *** FIREWALL ROUND 8: [...] ***`), and `analyze_matches.py` replays them through the engine — [tournament-matches.json](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/matches/tournament-matches.json), [STATEMENT.md](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/STATEMENT.md).

### Inferences
- Regression-suite recipe (about 20 min to build):
  1. For each lost replay, step back from the loss event (kernel capture or firewall burn) to the last own decision where a safe alternative existed.
  2. Store (observation, colour, required property). Example properties:
     - "kernel not on the ring locked at the next firewall round, given actions left";
     - "after our turn, no enemy piece can capture the kernel within 2 actions".
  3. Run the candidate through the referee-equivalent MCTS (100 simulations) on each position and assert the property.
- Gate: pass 100% of the positions the incumbent passes, with no new failures. Also include winning positions, so that a fix doesn't make the model over-defensive.
- Refresh the replay export before each upload decision. The seat plays about 100-150 games per hour, so new losses accumulate quickly and show which blunders the current field exploits.

### Gaps
- Only our own replays are visible, so we can't build tests from other teams' blunders.
- The 5-versus-8 discrepancy in the firewall count comes from parsing (our kernel listed among the burnt pieces versus any firewall-decided loss); it was not resolved.

## Q6. Exploitation vs robustness, and the resulting deploy protocol

### Takeaway
Maximise general strength and remove systematic blunders. Don't model specific opponents: we see only our own games, the field is diverse, and matchmaking keeps changing who we meet. The cheap form of "exploitation" is to fix our own common blunders and make sure we punish the same blunders in others: kernel capture, the firewall lock, and the Magenta tiebreak.

### Cited Findings
- The field is wide. Top ratings on the dashboard snapshot are CVCT 2020, NicoSeverin 1943 and ByteM3 Pro 1899, with 127 active kernels in the last window, and several unnamed teams oscillated between 1100 and 1700 within 15-20 games — [tournament-dashboard.json](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/matches/tournament-dashboard.json).
- AlphaStar needed exploiters specifically to find and patch the main agent's weaknesses; pure self-play or PFSP was not enough for robustness — [DeepMind AlphaStar blog](https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/).
- Spinning-top geometry implies that at mid strength there are many cycles, so a model tuned against one style can lose to another — [Czarnecki et al. 2020](https://arxiv.org/abs/2004.09468).

### Inferences
- **Opponent modelling.** Use replays only to measure how often opponents fall into the firewall and kernel-capture traps, and make sure our search punishes them; hand_v2's capture-threat features already do. Don't tune toward any single named team: matchmaking rotates opponents as our rating moves.
- **Deploy protocol for the remaining 1-3 hours.** All evaluation runs on one CPU core.
  1. **Now:** if hand_v2 passes the regression suite (Q5) and a 10-game sanity check against random and greedy, upload it. The evidence is already strong (68/80 against stopgap, Wilson lower bound 0.756; 97% against seat_l3). Log it in DEPLOYS.md.
  2. **Watch** the first 20-30 live games (10-20 min) and log win rate per colour. Roll back if the live win rate stays ≤ 35% over 20 games while the rating is flat or falling.
  3. **Every further candidate** must pass, in this order:
     - the regression suite (seconds);
     - pair SPRT against the incumbent, H0 = 0.50, H1 = 0.65, α = β = 0.10, LLR bounds ±2.20, capped at 120 games (about 2 CPU-min); count distinct games;
     - 20-game gauntlet checks against seat_l3, az_v1/az_v2, hand_v1, and a rusher or turtle bot, with no more than a 10-point drop against any member;
     - per-colour reporting.
  4. **Deadlines:**
     - For the last upload, allow ≥ 30 min before the end for an edge of ≥ +100 Elo head-to-head, and ≥ 45-60 min for a smaller one.
     - No uploads in the last 15 min.
     - If nothing clears the SPRT at 0.65, keep the seat; a ±45-70 rating noise floor swamps a 55% edge.
  5. **Don't stack uploads** less than about 20 games apart. Each restart raises volatility and resets the 8-game display, and you lose the live signal you need to judge the previous upload.

### Gaps
- No primary-source data was found on what share of CCC teams submit AlphaZero-style versus hand-made models. The env ships only a random reference model, so there is no common "default" model to exploit.
- Lc0 and Kaggle-specific leaderboard practices could not be verified with primary sources within the time budget.

## Q7. Applying the contest scoring slide: post-upload rating distribution, upload cutoff, the 8-game display, one upload or several

### Takeaway
- An upload never lowers the expected rating if the model is truly better. It does widen the final-score spread: SD rises from about 45 when keeping the seat to about 55, 70 or 90 for mild, moderate or strong volatility bumps. In standard Glicko-2 that extra spread may last until the end.
- A +5 pp edge ends below the keep-the-seat counterfactual about one time in three. A +10 pp edge does so in 15-30% of runs, a +20 pp edge in under 10% once about 16 games have been played.
- Bundle improvements into one upload that clears "Wilson 95% lower bound > 0.55 against the current seat, at least 60 colour-paired games" (≥ 41/60), ideally ≥ 30 games before the end.
- Stop uploads about 60 min before the end for small or medium edges, and about 25-30 min before for large ones.
- Never let the contest end before the current model has completed 8 games; the slide is ambiguous about what is shown otherwise.

### Cited Findings
- The slide text, as written in our extract:
  - "Every game is rated with Glicko-2 ... Everyone starts at 1500."
  - "Early games move your rating a lot, later games much less."
  - "Your score appears on the contest leaderboard once your current model has played 8 games and is updated after every game after that."
  - "Uploading a new model keeps your rating but makes it swing more for a while and restarts the 8-game count. Upload when the new model is clearly better."
  - "Ranking: highest level first, then highest score. The score at the end of the contest counts."
  - "Upload your model early. Models submitted near the end of the contest play fewer games, so their final ranking may be less accurate."

  An earlier slide says "Your score reaches the contest leaderboard once your current model has played 8 games" — [level-4.txt, lines 56-57 and 72-83](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/level-4.txt).
- In Glicko-2, σ only changes through the τ-constrained volatility step. RD is inflated each period by σ′, as φ* = √(φ² + σ′²) — [Glickman, Glicko-2 example](http://www.glicko.net/glicko/glicko2.pdf). In our simulation with τ = 0.5 and one game per period, a raised σ barely decayed over 100 games: 0.10 stayed 0.10 and 0.25 stayed 0.25. RD therefore settles at a higher floor: about 78 at σ 0.10, 95 at σ 0.15 and 121 at σ 0.25, against about 61 at σ 0.06 (own calculation).
- **Rating change after an upload.** The new model's true strength is set to rating + edge, with edges of +5, +10 and +20 pp (≈ +35, +70, +147 Elo). Opponents sit within ±60 with RD 80; 600 runs per cell (own calculation, scratchpad `calc2.py`). Each cell shows mean change / SD / P(change < 0) / P(final rating below keeping the old seat):

  | scenario | N = 8 | N = 16 | N = 30 | N = 50 | N = 100 |
  |---|---|---|---|---|---|
  | keep seat (σ .06, RD 65), edge 0 | +1 / 28 | +0 / 36 | +1 / 40 | +2 / 43 | +0 / 45 |
  | U1 (σ .10, RD 65), +5 pp | +9 / 34 / .39 / .44 | +15 / 44 / .35 / .37 | +24 / 55 / .33 / .34 | +29 / 55 / .30 / .36 | +36 / 54 / .27 / .29 |
  | U1, +10 pp | +20 / 32 / .26 / .32 | +34 / 42 / .21 / .28 | +51 / 51 / .17 / .20 | +62 / 53 / .12 / .19 | +72 / 57 / .10 / .15 |
  | U1, +20 pp | +38 / 31 / .12 / .19 | +74 / 42 / .04 / .10 | +106 / 51 / .02 / .05 | +132 / 51 / .01 / .02 | +141 / 54 / .00 / .02 |
  | U1, 0 pp (no real gain) | +5 / 31 | +3 / 44 | +2 / 50 | +3 / 54 | +2 / 54 |
  | U1, −5 pp (hidden regression) | −10 / 33 / .63 / .60 | −19 / 45 / .64 / .63 | −28 / 53 / .69 / .65 | −32 / 54 / .72 / .68 | −37 / 55 / .75 / .72 |
  | U2 (σ .15, RD 120), +5 pp | +19 / 63 / .40 / .39 | +28 / 67 / .36 / .37 | +37 / 66 / .27 / .30 | +34 / 67 / .30 / .34 | +35 / 64 / .30 / .33 |
  | U2, +10 pp | +40 / 67 / .28 / .29 | +54 / 71 / .24 / .27 | +65 / 67 / .16 / .20 | +69 / 68 / .14 / .19 | +69 / 68 / .16 / .20 |
  | U2, +20 pp | +79 / 63 / .11 / .12 | +113 / 66 / .05 / .06 | +136 / 68 / .01 / .05 | +148 / 71 / .02 / .04 | +147 / 66 / .02 / .03 |
  | U3 (σ .25, RD 200), +5 pp | +18 / 102 / .43 / .44 | +25 / 91 / .39 / .38 | +34 / 89 / .35 / .34 | +37 / 84 / .33 / .36 | +35 / 87 / .35 / .36 |
  | U3, +10 pp | +51 / 96 / .30 / .32 | +65 / 92 / .24 / .27 | +68 / 85 / .21 / .23 | +70 / 87 / .20 / .24 | +70 / 89 / .22 / .23 |
  | U3, +20 pp | +111 / 96 / .10 / .12 | +138 / 93 / .07 / .08 | +150 / 96 / .06 / .08 | +149 / 85 / .04 / .06 | +147 / 88 / .03 / .05 |

  The server's actual post-upload σ and RD are unknown; U1-U3 bracket "swings more for a while".
- **Game rate in our replays.** The 17 inter-game gaps between 13:35:02 and 13:42:15 UTC have a mean of 25.5 s, a median of 22 s and a maximum of 81 s, which is about 140 games per hour. The dashboard's global rate (100.4 games/min, 127 active seats) implies about 95 per seat per hour — [tournament-matches.json](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/matches/tournament-matches.json), [tournament-dashboard.json](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/matches/tournament-dashboard.json). This is a single 7-minute window from the tournament's first minutes; no later replays have been exported.
- **8-game display, dashboard evidence.** In the tournament dashboard, rows whose current model has played fewer than 8 games are still listed and ranked with their kept rating:
  - NoClueCrew: rank 8, elo 1797, `games: 0`, `win_rate: null`, 275 overall games per colour;
  - an UNNAMED TEAM: rank 3, elo 1937, `games: 1`;
  - SQLExceptionPhil: rank 9, `games: 1`;
  - Trei patrate: rank 7, `games: 6`.

  The `games` field counts the current model's games, while `overall_*` fields count all games — [tournament-dashboard.json](C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/matches/tournament-dashboard.json).

### Inferences
1. **Expected change and variance after an upload.** In expectation, the rating moves toward the new model's true strength (+35, +70 or +147 for +5, +10 or +20 pp). With the raised σ it gets there in about 16-30 games, against about 50 games at σ 0.06; the SD of the change over N ≥ 30 games is about 55, 70 or 90 for U1, U2 or U3.
   - **The extra swing is a net risk** when the true edge is no larger than the added noise: √(SD_upload² − SD_keep²) ≈ 30 for U1, 50 for U2 and 75 for U3. A +5 pp edge (+35) is at or below that in every scenario, so P(worse than keeping) stays at 0.29-0.44.
   - It is also a net risk when few games remain (N ≤ 8-16), and whenever the edge is not really positive: a 0 pp upload is a pure variance add, and −5 pp loses 60-75% of the time.
   - For a risk-neutral team (maximising expected final score) any true positive edge is worth uploading. A team protecting a narrow gap to the next rank should upload only edges of ≥ +10 pp, with N ≥ 30 remaining.
2. **Cutoff.** Plan with a conservative 90 games per hour (1.5 per minute), below both observed estimates, since the rate late in the contest is unknown.
   - A +20 pp edge needs about 16-30 games to realise ≥ 75-90% of its gain, so the last upload should be at T−25-30 min.
   - A +10 pp edge needs about 30-50 games, so T−40-60 min.
   - A +5 pp edge never clearly beats keeping, so don't upload it in the last 60-75 min (and arguably not at all).
   - Hard stop for everything except an emergency rollback of a broken seat: T−15 min. That leaves the 8-game count about 5 min at 90 per hour, plus margin for a slowdown.
   - The slide's own warning ("models submitted near the end ... play fewer games, so their final ranking may be less accurate") points the same way.
3. **8-game display ambiguity.**
   - What is clear: the score appears only after 8 games of the *current* model, and an upload restarts that count.
   - What is unclear:
     - whether the official contest leaderboard, during games 1-7 of a new model, shows nothing, shows the frozen pre-upload score, or shows the live rating;
     - whether "the score at the end of the contest counts" means the displayed leaderboard score or the internal rating;
     - whether a hidden score at the end could even drop the team to "Level 3 reached", since ranking is "highest level first".
   - The tournament dashboard JSON keeps listing sub-8-game teams at their kept rating (NoClueCrew at 0 games), which hints that the rating persists internally. But that is the tournament dashboard, not necessarily the contest leaderboard.
   - So never let the contest end with fewer than 8 games on the current model, and ask the organisers if possible.
4. **One combined upload or several.**
   - If at least about 50 games remain after the last upload, the final rating depends mainly on the *final* model's true strength. Intermediate uploads only move the starting point closer, add live replays, and each one re-raises σ, which in standard Glicko-2 may persist and widen the final spread.
   - Verification cost favours bundling. A bundled +15-20 pp candidate passes a 0.70 SPRT in about 20-35 games; three +5 pp increments would each need about 350-530 games and can't be verified in the time left.
   - So: deploy early when a candidate is clearly better (it climbs and gives replays). After that, batch further improvements into one gated candidate, with at most one more upload in the last 60 min.
   - **"Clearly better" in local arena terms**, against the current seat, with colour-paired games (each opening seed played with both colours, more than 70% of games distinct). All of the following must hold:
     1. SPRT H0 = 0.50 vs H1 = 0.65, α = β = 0.10 (LLR ±2.20), accepted within a cap of 120 games; or fixed-n with Wilson 95% lower bound > 0.55, meaning ≥ 41/60, ≥ 53/80 or ≥ 65/100.
     2. In the last 60 min, the stricter bound > 0.60: ≥ 44/60 or ≥ 57/80.
     3. Neither colour below 50% against the incumbent.
     4. No gauntlet member down more than 10 pp.
     5. ≥ 95% against random and greedy.
     6. 100% of the regression positions pass.
   - hand_v2 (68/80, lower bound 0.756, 82% as CYAN and 88% as MAGENTA) clears even the strict version.

### Gaps
- The server's τ, post-upload σ and RD, and whether volatility is actively decayed ("for a while") are unpublished. The tables bracket them but don't pin them down.
- Whether a fewer-than-8-games seat counts at the end, and what the contest leaderboard (as opposed to the tournament dashboard) shows meanwhile, is not documented. Only organisers can resolve this.
- Late-contest game throughput is unknown; the 90/h planning figure is an assumption below both observed estimates.
