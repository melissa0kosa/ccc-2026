# Ship exact endgame knowledge, not more tricks

Across the three specialty fields and the five neighbouring games, only one family still pays before the 16:20 UTC cutoff: **exact low-material endgame knowledge**, delivered either as training data or as a hard-coded rule. The team trained **az_res4f_v3 at 15:32** with nearly every plateau-breaking change the research converged on: pre-trade and Trace-vs-piece starts, live-game seeds, outcome-only values in low material, a doubled deep-endgame value weight and a OneCycle learning rate. So the best gain per hour is now **gating v3 against the live az_res4f_v2**. Second comes the **tablebase-style bare-Kernel override (rule (a))**, which changes nothing outside its gate and should ride on whichever model ships. The small-net deep-learning tricks have been measured and are spent. **2-, 4- and 8-symmetry test-time ensembles scored 52%, 55% and 51%** against the seat, value scaling peaked at a non-passing 58.8%, and the v20+v30 ensemble scored 47.5%. The neighbouring games (King of the Hill, Racing Kings, Fairy-Stockfish extinction, Ludii, Breakthrough) mostly tell the team what *shape* the hand base's race and guard terms should take. Acting on that needs a base refit plus a residual retrain, so it belongs after the contest. The binding constraint before 16:20 is the gate itself. At **46–48 games per hour** and a Wilson lower bound of 0.55, a true 58% candidate passes **only 13% of the time at 220 games**, and the team's own bar rises to 0.60 after 16:00. After the contest, the highest-value project is a **finite-horizon endgame tablebase**: 13.7 million K+X vs K states on the 6×6 phase, solved by plain backward induction because the round counter makes the game graph acyclic. Its results would then rescore training labels the way Lc0 does.

## Endgame labels broke the plateau, and v3 already carries the rest

Every recipe change on the old positions left the head-to-head against az_res_v20 flat. **Twice the steps (v30) scored 53%**, a heavier search-Q share in the value target (v31) 46%, 500 extra v20 games (v32) 48%, and the new hand_v4f base without new data (az_res4f_v1) 50% in the az arena and 54.1% in the gate ([az RESULTS.md](../agents/az/RESULTS.md), [gate/RECOMMEND.md](../gate/RECOMMEND.md)). Adding endgame-start self-play broke it: 4000 games of Kernel plus at most one piece per side, replicated to **26% of samples**. az_res4f_v2 scored **144/220 = 65.5% [59.0, 71.4]** in the gatekeeper's arena and has held the seat since 15:19 ([DEPLOYS.md](../DEPLOYS.md)). The v2 data set also gained fresh middlegame games, but v32 shows that fresh games alone do nothing, so the endgame starts are the lever. Self-play research has seen this pattern before. KataGo's cyclic-group blind spot did not fade with more self-play. It went away when **0.08% of self-play games** started from hand-written blind-spot positions, after which a May 2023 net beat the original adversary in **94.1%** of games without search ([FAR AI](https://goattack.far.ai/positional-adversarial-training)). New adversaries still found new holes ([Tseng et al. 2024](https://arxiv.org/pdf/2406.12843)), and the same should be expected here. Once the bare-Kernel family is fixed, the next blind spot should sit one phase earlier, at the decision to trade into low material.

Outcome-only labels in low material are the correct target, not a compromise. The review's "truth" is the game outcome plus 800-simulation searches by the same models, so the 100- and 800-sim searches agree on the losing endgame moves. Search Q therefore carries the value net's blind spot straight back into training ([reviews/az_res_v6.md](../reviews/az_res_v6.md)). That is consistent with v31, which used more Q, doing worse. It is also why the coordinator dropped hard-example mining from 100- vs 800-sim disagreements.

The seat still misjudges Magenta endgames. On the 299 live low-material positions, az_res4f_v2 reaches **sign accuracy 0.79, MSE 0.487 and a mean value of −0.30 in positions Magenta later lost**, against 0.71, 0.704 and −0.22 for v20 ([gate/RECOMMEND.md](../gate/RECOMMEND.md)). The review's pass criterion asks for that Magenta-lost mean to fall below −0.4 ([reviews/az_res_v20.md](../reviews/az_res_v20.md)). The seat misses it, so the K+1 endgame lever is not exhausted.

The code shows how far the team has already pushed it. At 15:33 the repo had changed since the coordinator's 15:27–15:32 check, which found a constant learning rate. `sp_end.py` now generates three start types: k01 (K + 0–1 piece, rounds 9–24), kt1 (K+Trace vs K+1, rounds 6–16) and k2 (K+2 pre-trade, rounds 6–16) ([sp_end.py](../agents/az/sp_end.py)). `gen_end2.py` adds every position from the v20 live games with at most two non-Kernel pieces per side and replays each one several times ([gen_end2.py](../agents/az/gen_end2.py)). That is the "average several playouts per seed" label-noise fix. `azlib.py` uses outcome-only values wherever each side has at most two non-Kernel pieces, and doubles the value loss at one piece or fewer ([azlib.py](../agents/az/azlib.py)). `retrain_fast.py` gained a OneCycle schedule and an SWA path ([retrain_fast.py](../agents/az/retrain_fast.py)). The v3 run used 3000 endgame games replicated to **30% of 373,322 positions** with OneCycle, and exported at 278 nodes / 321 initializers with the validator accepting it ([az log.txt](../agents/az/log.txt)).

Two caveats matter for the gate. First, `games_end_train2.pkl` was written at 15:30, after az_res4f_v2 existed, but the log does not record which model generated it. Second, the az agent's in-loop arena measured v2 at 76% against v20 while the gatekeeper measured 65.5%, a gap of about 2.3 standard errors. Only the gatekeeper's arena should judge v3.

## Tablebase practice says hard-code the one exact layer

Classic endgame solving runs backward from terminal positions. It generates predecessors with an un-move generator and iterates until no new positions appear ([CPW: Retrograde Analysis](https://www.chessprogramming.org/Retrograde_Analysis)). At scale this produced Chinook's checkers databases, the largest covering **39 trillion positions** ([Schaeffer et al. 2007](https://www.cse.sc.edu/~mgv/csce580sp17/gradPres/schaeffer_CheckersIsSolved_Science2007.pdf)). Syzygy shows that a clock can change the value of identical boards: its 50-move counter makes some wins "cursed" ([python-chess Syzygy docs](https://python-chess.readthedocs.io/en/latest/syzygy.html)).

COLONY: ONE is easier in one way. The round counter rises strictly to round 26, so the game graph is a **finite DAG layered by round, side and actions left**. Solving it is plain backward induction over at most 72 half-turn layers for rounds 9–26, with no fixpoint and no un-move generator. Counting ordered placements on the 6×6 phase gives **428,400 K+X vs K states per layer (13.7 million across 32 layers)** and **35.3 million K+X vs K+Y states per layer (1.13 billion)**. The rings are concentric and no piece has a forward direction, so the eight board symmetries are valid and divide these counts by about 8. Colour is not a symmetry, because Magenta wins ties ([endgame scratch analysis](file:///C:/Users/melis/AppData/Local/Temp/claude/C--Users-melis-Downloads-cloudflight-coding-contest/2537b5b1-8fcc-4ac8-8fd3-b627f7b7b210/scratchpad/eg_stats.py)).

The obstacle is the move generator, not the size. The environment is pure Python, and its docs say not to reimplement the rules ([observation.md](../data/colony-one-env/docs/observation.md)). At an assumed 10 µs or more per step, enumerating even K+X vs K takes at least 35 single-core minutes. A vectorised generator would have to reproduce the bodyguard order, Trace blocking and Exploit jumps exactly, which is a post-contest job.

One layer needs no table: the bare side with one action left. The measurements locate the error precisely ([eg_val2.py](file:///C:/Users/melis/AppData/Local/Temp/claude/C--Users-melis-Downloads-cloudflight-coding-contest/2537b5b1-8fcc-4ac8-8fd3-b627f7b7b210/scratchpad/eg_val2.py)). On average the nets are calibrated for bare Kernel vs K+1: v20 −0.65, az_res4f_v2 −0.63, empirical −0.62. They fail in one sub-configuration: the bare side has one action, its Kernel is adjacent to the enemy's last piece, and that piece sits within Chebyshev 2 of the enemy Kernel, so capturing it is suicide by the bare-Kernel rule. There, **v20 says −0.52 to −0.62 and az_res4f_v2 −0.56, against an empirical −0.83 to −0.90**, and 12–20% of positions get the wrong sign. The same pattern decided live games. At 10773, 11686 and 12113 action 40, Magenta had one action and every move lost. v20 nonetheless valued the position at +0.34, +0.56 and +0.41 and put a 0.93–0.95 prior on the suicide capture. The v4f-based net gets the sign right (−0.45 to −0.57) but not the magnitude ([reviews/az_res_v6.md](../reviews/az_res_v6.md)).

Rule (a) went to the team in a fixed check order. First, the value is +1 if we can capture an unguarded enemy Kernel. Second, it is +1 if we can capture the last enemy piece and land at least 3 squares from their Kernel on a smaller ring (an equal ring is enough as Magenta). Otherwise it is about −0.95. That fallback is empirical: in 100-sim self-play, the side with K+1 against a bare Kernel wins **97.7% (Cyan to move, n = 3290) and 98.9% (Magenta, n = 2797)** ([eg_stats.py](file:///C:/Users/melis/AppData/Local/Temp/claude/C--Users-melis-Downloads-cloudflight-coding-contest/2537b5b1-8fcc-4ac8-8fd3-b627f7b7b210/scratchpad/eg_stats.py)). Those positions are correlated within games, so the effective sample is smaller.

The rule is **gated off when we are Magenta and the firewall-urgency plane equals 1.0**. In that case the shrink fires right after our turn and can burn pieces, so the −0.95 fallback no longer holds. Captures are still safe to score, because bare Kernels are resolved before the firewall in `_turn_end` ([mcts.py](../data/colony-one-env/colony_one/mcts.py)). The distance-2 rule is checked before the ring comparison ([board.py](../data/colony-one-env/colony_one/board.py)).

The team built `az_res4f_v2_eg.onnx` at 15:34, with a verifier that checks bit parity outside the gate and compares the gated cases against env brute force, including the Magenta-urgency gate-off ([eg_verify.py](../agents/hand/eg_verify.py)). Its effect will be small but targeted. With 100 simulations, the search reaches these leaves from one turn earlier and stops walking into the suicide line. It does not fix the earlier K+guard vs K+X misjudgements.

This is a hard-coded version of Stockfish NNUE's material buckets, which pick one of 8 layer stacks by piece count ([Stockfish-derived network.cpp](https://huggingface.co/spaces/Amogh1221/deepcastle-api/blob/main/src/nnue/network.cpp)). Inside its gate the "expert" is exact rather than learned. The next step is the Lc0 rescorer, whose notebook says rescoring "is used in the training of all the best nets for leela, and should always be used before running training" ([lc0 Colab mirror](https://science.ecosyste.ms/projects/175905)). University of Alberta tests found that even strong Lc0 nets make measurable errors in endgames where exact tables exist ([Haque & Müller 2021](https://webdocs.cs.ualberta.ca/%7emmueller/ps/2021/2021_Haque_ACG_Road_to_Perfection.pdf)).

For this game, rescoring can start without any table. Replace the outcome label with the exact rule-(a) value on the roughly 2.7k gated training positions already on disk. Add the analogous K vs K one-action gate (gate B), then the full table.

## Neighbouring games fix the shape of the race and guard terms

King of the Hill and Racing Kings engines score a race to a goal region with **steep lookup tables, not linear terms**. Multi-variant Stockfish uses `KothDistanceBonus = {1949, 454, 151, 75, 42, 0}`, roughly 4×, 3× and then 2× per step, with almost equal midgame and endgame values. It counts each enemy attacker on a target square as an extra step. Its Racing Kings table starts at 14,282 and treats a blocked rank as one more step ([ddugovic Stockfish evaluate.cpp](https://github.com/ddugovic/Stockfish/blob/master/src/evaluate.cpp)). Fairy-Stockfish's capture-the-flag term is `4000·wins/(wins + dist²)`. Its search admits attacked squares one iteration late, because "a move will be needed to block or capture the attack" ([Fairy-Stockfish evaluate.cpp](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/master/src/evaluate.cpp)).

The team's hand_v4f base does neither. Its `kdiff` term is linear in the ring-depth difference, and its bare gate fires only when no non-Kernel piece is left ([fast4.py:245-246](../agents/hand/fast4.py)). Our own outcome table shows the step those engines encode. With one piece each, the side to move wins **0.41 / 0.48 / 0.74 / 0.81 / 0.89 as Magenta and 0.37 / 0.43 / 0.60 / 0.70 / 0.78 as Cyan**, for Kernel-depth differences of −2 to +2. The jump from −1 to 0 is **0.26 for Magenta and 0.17 for Cyan, two to four times the neighbouring steps** ([eg_stats.py](file:///C:/Users/melis/AppData/Local/Temp/claude/C--Users-melis-Downloads-cloudflight-coding-contest/2537b5b1-8fcc-4ac8-8fd3-b627f7b7b210/scratchpad/eg_stats.py)). That jump is the Magenta tie rule. One-hot buckets of {≤−2, −1, 0, +1, ≥+2}, with the zero bucket signed by colour, fit it; a single linear weight cannot.

Fairy-Stockfish's extinction variants supply the guard term. A threat on a piece class counts **1000/denom²**, where denom is that side's remaining count of the class above the extinction count, so a threat on the last piece weighs 4× a threat when two remain. Its pseudo-royal king-danger form is `20·attacks/(evasions + 1)`, applied quadratically ([Fairy-Stockfish evaluate.cpp](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/master/src/evaluate.cpp), [variants.ini](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/master/src/variants.ini)). Under our bodyguard rule, a lone guard is pseudo-royal: losing it leaves K vs K+1, which the piece side wins about 98% of the time. So guard danger should grow with enemy attacks and shrink with the guard's safe squares, not enter as a flat "hanging" term. Fairy-Max does not even know that a bare king loses in Shatranj, and a separate engine, ShaMax, was written for that rule ([fairymax(6)](https://manpages.ubuntu.com/manpages/noble/man6/maxqi.6.html)). That confirms baring rules need dedicated logic.

Stockfish's material-keyed endgame evaluators show how to wire such terms in without touching other positions: dispatch by material key, as in KXK's push-to-edge and push-close ([Stockfish sf_15 endgame.cpp](https://github.com/official-stockfish/Stockfish/blob/sf_15/src/endgame.cpp)). For us, add a gated low-material term inside the tanh, so every other position keeps bit-exact parity. The push-close analogue has the opposite sign here, because Kernels within distance 2 lose for the side that just finished its turn.

Two more transfers concern search and the prior. Ludii biases PUCT with a linear softmax over action-anchored local patterns, keeps about **15 features after pruning**, and needs a few hundred self-play games. Its greedy linear policy alone reached **40% against UCT** in Breakthrough and Teeko ([Soemers et al. 2019](https://ar5iv.labs.arxiv.org/html/1903.08942)). For us that would be a logit correction for patterns like the recapture trap and a lone guard stepping into a shot line, fitted to 800-sim visits. It has never been tested on top of a CNN prior.

Implicit minimax backups "increase the strength of MCTS significantly" in Breakthrough, Kalah and Lines of Action, with α of 0.15–0.4 a safe choice ([Lanctot et al. 2014](https://arxiv.org/abs/1406.0486)). The referee's search is fixed, but our self-play search is not, so minimax backups there would sharpen Q targets in forced endgame lines.

The remaining games add confirmation rather than new tools:

- **Onitama:** a full retrograde solve was shown infeasible, a forward-looking retrograde analysis weakly solved the game, and a board symmetry halved the work ([Leiden thesis](https://studenttheses.universiteitleiden.nl/access/item%3A4262355/view)). That supports a material-restricted table over a full one.
- **Santorini:** the AlphaZero fork prefers a smaller net with more simulations ([cestpasphoto](https://github.com/cestpasphoto/alpha-zero-general)). That cannot apply here, because the simulation count is fixed at 100.
- **CodinGame:** the strong bots use MCTS with hand evaluations, and one AlphaZero-style pipeline "would often collapse into both sides WAITing" ([CodinGame forum](https://forum.codingame.com/t/spring-challenge-2021-feedbacks-strategies/190849?page=4)). This echoes our passivity losses; nothing else transfers.

All of these changes land in the hand base, and the residual is trained on the base's features. So each one costs a base refit, a residual retrain and a gate. The last two base attempts show the risk. hand_v5's shot-aware maps scored **46% and 50%** against v4f, and prior-only changes without a value refit fell to 38% ([hand RESULTS.md](../agents/hand/RESULTS.md)). That full cycle does not fit before 16:20.

## Symmetry ensembles and weight averaging are measured and spent

The literature supports test-time symmetry averaging. AlphaGo Zero evaluated each position under a random rotation or reflection ([AlphaZero paper](https://arxiv.org/pdf/1712.01815)), and KataGo averages a sampled subset of the 8 symmetries at the root ([Tseng et al. 2024](https://arxiv.org/pdf/2406.12843)). Neither publishes an Elo figure for it.

Locally the idea looked promising because the residual is far from equivariant. Two-symmetry averaging changes the prior's top move in 24% of endgame positions, at **2.41 ms vs 1.59 ms** (minimum latency per batch-8 call) for the original ([tta_soup.py](file:///C:/Users/melis/AppData/Local/Temp/claude/C--Users-melis-Downloads-cloudflight-coding-contest/2537b5b1-8fcc-4ac8-8fd3-b627f7b7b210/scratchpad/tta_soup.py)). The team then measured it against the seat. The two-symmetry version scored **52/100**. The four-symmetry version scored 55/100, with Magenta at 44%, used 3.10 s of CPU per game and was rejected by the validator as "too slow" three times. The eight-symmetry version scored 27/53 at 6.53 s ([gate/RECOMMEND.md](../gate/RECOMMEND.md)). A large change of unknown sign turned out to be neutral.

Weight averaging needs a shared basin. In model soups, a greedy soup reached 81.03% top-1 against 80.38% for the best single model, while a uniform soup fell to 79.97% ([Wortsman et al. 2022](https://arxiv.org/html/2203.05482v3)). SWA lifted a pretrained ResNet-50 from 76.15% to 76.97% after a BatchNorm recompute ([Izmailov et al. 2018](https://ar5iv.labs.arxiv.org/html/1803.05407)).

The team's checkpoints behave the same way ([tta_soup.py](file:///C:/Users/melis/AppData/Local/Temp/claude/C--Users-melis-Downloads-cloudflight-coding-contest/2537b5b1-8fcc-4ac8-8fd3-b627f7b7b210/scratchpad/tta_soup.py)). The midpoint of v20 and its own continued run v30 beat both ends on held-out policy loss (2.0756). The midpoint of v20 and v32, which saw a different batch order, was worse than both (2.0925). Either way the differences are about 0.005 nats, and the output ensemble az_res_ens2 (v20+v30) screened at **47.5%** ([gate/RECOMMEND.md](../gate/RECOMMEND.md), [ens.py](../agents/az/ens.py)). SWA along v3's own trajectory is safe and free at inference. The code path exists, but v3 was trained without it, so it could at most run as a sibling candidate in an idle arena slot.

The other small-net levers are closed for today:

- **Value scaling** was dropped by the coordinator: stopgap ×2 scored 32% and hand v3c ×1.3 scored 49%. The round-4 variants confirm it. Value temperature 1.25 scored 52%, and residual-value ×1.25 reached **58.8% [52.4, 64.8]** but failed the gate and lost 5 points against stopgap ([gate/RECOMMEND.md](../gate/RECOMMEND.md)).
- **Distillation** compresses an ensemble into one net ([Hinton et al. 2015](https://arxiv.org/abs/1503.02531)), but the team's distilled students plateaued at 42–54% against v20.
- **Auxiliary heads** cost KataGo **1329 → 1139 Elo** when its ownership and score targets were removed ([Wu 2019](https://ar5iv.labs.arxiv.org/html/1902.10565)). They pay back over long runs, not one retrain.
- **Value-head capacity** is not the bottleneck. The same 64-unit head cut endgame MSE from 0.392 to 0.340 (Magenta) and from 0.413 to 0.320 (Cyan) once endgame data arrived ([az RESULTS.md](../agents/az/RESULTS.md)).

## The gate, not the model, decides what reaches the seat

The arithmetic has changed since the first report. The live game rate is **0.76–0.8 games per minute (about 46–48 per hour)**, not 95, and the "two forfeits per upload" were a download artefact ([DEPLOYS.md](../DEPLOYS.md)). At a Wilson lower bound above 0.55, a true 58% candidate passes **about 13% of the time at 220 games**, because it needs an observed 61.7% or more. At a lower bound above 0.50 with 300 games, the same candidate passes **81%** of the time ([plateau_gate.py](file:///C:/Users/melis/AppData/Local/Temp/claude/C--Users-melis-Downloads-cloudflight-coding-contest/2537b5b1-8fcc-4ac8-8fd3-b627f7b7b210/scratchpad/plateau_gate.py)). In a Glicko-2 simulation of an +8-point edge, uploading with 30 games left gains **+43 on average (SD 75), and ends worse than keeping the seat in 28% of runs**; with 60 games left the figures are +51 and 24% (same script).

The team's own rules make the window narrow:

- no +10-point upload after 15:50;
- a 0.60 lower bound after 16:00, which in practice means about 68% over 220 games;
- no +20-point upload after 16:20;
- about 20 games between uploads, so the slot after the 15:19 upload opened around 15:44 ([DEPLOYS.md](../DEPLOYS.md), [gate/RECOMMEND.md](../gate/RECOMMEND.md)).

So v3 has to be screened now. If it clears 0.55 before about 15:50 it ships. Between 16:00 and 16:20 it ships only with a 0.60 lower bound.

Relaxing the gate cuts both ways, and round 4 shows exactly how. Under a lower bound above 0.50, the residual-value ×1.25 variant would have passed on its head-to-head (lower bound 52.4), even though the coordinator had dropped value scaling on earlier evidence. That small, sign-uncertain kind of edge is what a relaxed gate admits. Head-to-heads between siblings are also non-transitive here: distill_v3 scored 76.7% against hand_v2 but 60% against az_res_v6 ([gate/RECOMMEND.md](../gate/RECOMMEND.md)). AlphaStar's league and its reimplementations treat this as the reason to evaluate against a population ([TStarBot-X](https://arxiv.org/pdf/2011.13729)).

Whatever threshold the team picks, it should therefore keep three guards:

- the gauntlet;
- the 299-position low-material suite, which the round-4 table already runs;
- a one-minute check of held-out loss and calibration before any arena is spent.

## Ranked by gain per hour, with status

The pipeline has six parts. Each has a clear best move on each side of the cutoff:

| Pipeline part | Best before 16:20 | Best post-contest | Field it comes from |
|---|---|---|---|
| Hand-feature base | Nothing new (a base change needs a full refit and retrain cycle) | One-hot race buckets, attacked-path increments, pseudo-royal guard danger | KOTH, Racing Kings, Fairy-SF extinction, SF endgames |
| Residual net | Gate v3; optionally an SWA sibling | Material-gated endgame head; TTA-S8 distillation | NNUE buckets, SWA, distillation |
| Self-play / training data | Already in v3 (kt1, k2, live seeds) | Trade-decision seeds and exploiter-found positions | KataGo positional training, AlphaStar |
| Value targets | Already in v3 (outcome-only in low material, ×2 weight) | Tablebase rescoring; implicit-minimax self-play Q | Lc0, Breakthrough/LOA |
| Endgame play | Rule (a) override stacked on the shipped model | Full K+X vs K and K+X vs K+Y tables | Retrograde analysis, Syzygy |
| Evaluation / gating | v3 gate on the gatekeeper's arena plus low-material suite; gate-threshold decision | Population (PFSP) evaluation | Fishtest statistics, AlphaStar |

**Before 16:20.** Gains are estimates, not measurements.

| Rank | Item | Part | Team time | Expected gain | Status |
|---|---|---|---|---|---|
| 1 | Gate az_res4f_v3 vs az_res4f_v2 on the gatekeeper's arena (≥220 games, both colours), plus lowmat.py and calib.py | Data, value targets | 10–15 min arena | Largest available: the only candidate with new information (4f_v1 → v2 gained 11 points in the gate) | Already done by team (trained 15:32); gate pending |
| 2 | Rule (a) exact override plus suicide-capture veto, stacked on whichever model ships | Endgame play | 10–20 min to finish verification | Small but exact; zero change outside the gate; cannot pass the gate alone | Sent for testing (v2_eg built 15:34) |
| 3 | Gate threshold: lower bound > 0.50 at ≥300 games plus gauntlet before 16:00, strict after | Evaluation | 0 min | Raises the pass rate of a true 58% from 13% to 81% | Sent for testing (DEPLOYS.md still at 0.55/0.60) |
| 4 | Pre-arena screen: held-out loss, calib.py, low-material suite | Evaluation | 1–5 min per candidate | Saves arena slots | Low-material suite already done by team; loss screen sent for testing |
| 5 | SWA tail sibling of v3 (about 1500 steps, BN recompute) | Residual | ~2 min GPU plus an arena slot | 0–2 points | Already done by team (code path exists; not used in v3) |
| 6 | Gated low-material race/guard term inside the tanh (KOTH buckets, Magenta-signed tie bucket) | Hand base, value | 35–45 min plus gate | Unknown; only low-material games | Sent for testing; realistically post-contest |

**Settled or dropped.**

| Item | Status |
|---|---|
| In-graph D4 test-time averaging (2/4/8 symmetries) | Already done by team: 52%, 55% (too slow, Magenta 44%), 51% |
| Value pre-tanh scaling or temperature | Dropped (stopgap ×2 = 32%, hand v3c ×1.3 = 49%; round-4 variants 52% and 58.8%, both failed) |
| Output ensembles or soups of v20/v30/v31/v32 | Dropped (ens2 47.5%; soups move held-out loss by about 0.005 nats) |
| Hard-example mining from 100- vs 800-sim disagreements | Dropped (the 800-sim search shares the bias) |
| K+2 pre-trade and K+Trace vs K+1 starts in rounds 6–16 | Already done by team (sp_end.py k2, kt1) |
| Live-loss seeds with repeated playouts | Already done by team (gen_end2.py) |
| Regenerate endgame data after az_res4f_v2 | Already done by team (games_end_train2.pkl at 15:30; generating model not logged) |
| Outcome-only values in low material, ×2 deep-endgame value weight | Already done by team (azlib.py) |
| LR annealing | Already done by team (OneCycle in v3; added after the coordinator's 15:27–15:32 check) |
| Ablation without endgame data | Dropped (v32 already shows fresh games alone are flat) |
| "Our last piece was a Trace in all 6 losses" | Dropped (false: game 11362 ended K+Daemon vs K+Trace, with our Daemon) |
| KL-to-previous-policy, alpha-beta teacher games, larger trunk | Dropped (the frozen base already anchors the policy; distill students 42–50%; no evidence the trunk binds) |
| KataGo surprise-weighted sampling | Sent for testing; not in v3 (uniform sampling), so effectively post-contest |

**Post-contest.**

| Rank | Item | Field | Why it ranks here |
|---|---|---|---|
| 1 | Finite-horizon tablebase: K+X vs K (13.7M states before symmetry), then K+X vs K+Y (1.13B), using a vectorised generator cross-checked against the env | Endgame solving | Exact labels exactly where search Q is blind |
| 2 | Lc0-style rescoring of all low-material labels, gate B, a material-gated endgame head | Lc0, NNUE | Rule-(a) rescoring of about 2.7k positions needs no table and can start first |
| 3 | Seeds at the trade-into-low-material decision, plus a main exploiter whose wins become seeds | KataGo, AlphaStar | Targets the predicted next blind spot |
| 4 | Hand base refit with KOTH-shaped buckets, attacked-path dilation and pseudo-royal guard danger; then re-base the residual | KOTH, Racing Kings, Fairy-SF | Matches the measured step at the tie boundary |
| 5 | Implicit-minimax backups in our self-play search | Breakthrough/LOA | Sharper Q in forced lines without touching the referee |
| 6 | Ludii-style linear action-feature prior correction | Ludii | Cheap to fit, but untested on top of a CNN prior |
| 7 | Surprise weighting, playout-cap randomisation, ownership-style auxiliary head, Reanalyse, 200-sim endgame labels | KataGo, MuZero | Proven at scale; pay back over many generations |
| 8 | Distil an 8-symmetry teacher, or add a symmetry-consistency loss | Small-net DL | Capped by the teacher; the 8-symmetry teacher itself scored 51% |

## Conclusion

The specialty fields agree on one point the team's numbers already showed. In an evaluator-only contest with a fixed 100-simulation search, gains came only from **new, exact information about positions the search cannot see**, never from averaging, ensembling or rescaling what the net already knows. Endgame-start data moved the seat by 11 points in the gate. No symmetry, soup, scaling or temperature variant reached a passing lower bound, and most landed within 5 points of parity. This game also has an unusual property: the round clock makes it a finite DAG, so its endgames are cheaper to solve exactly than chess's. Its tablebase should therefore be treated as core training infrastructure, not a luxury.

For the remaining 45 minutes, gating v3 matters more than any remaining modelling idea, and the threshold is a team choice of objective rather than a statistical fact. Under the current 0.55/0.60 rules, a genuinely better v3 is more likely to be rejected than accepted unless it is strongly better. If v3 ships, rule (a) should be stacked on it. If it fails, az_res4f_v2 with rule (a) keeps the seat, and the next leap waits for the tablebase.
