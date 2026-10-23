# Level 4: Tournament Time (COLONY: ONE)
Task:            Live tournament vs all other teams. One model "seat" per team; upload replaces it.
                 Each pairing: one game as CYAN and one as MAGENTA, mostly vs similarly rated teams, Level 4 scene.
Inputs:          level-4_scene.json (installed as data/colony-one-env/colony_one/scene.json); engine from Level 3 zip.
                 Pieces: Ping, Daemon, Exploit, Trace, Ice, Kernel (one of each per team).
Output format:   model.onnx: input observation (B,16,8,8) f32 -> policy (B,4096) logits, value (B,) in [-1,1].
                 Opset 13-18, <=5 MiB, allow-listed ops only, NO Constant op (c1.fix_constants). Referee MCTS 100 sims.
Metric:          Glicko-2 rating (start 1500). Ranking: highest level, then score. Score at contest end counts.
                 Leaderboard shows score after 8 games of the current model; new upload restarts the count and
                 raises volatility -> upload only when clearly better.
Hints/gotchas:   Magenta wins all tiebreaks (both kernels burnt, same-ring bare-kernel endgame). Global CYAN win
                 rate ~38%. Firewall locks the outer ring after rounds 8/16/24/26. Our live losses: kernel left on
                 the outer ring at round 8 (8/18 games), unguarded kernel captured early by Ping/Ice/Exploit/Daemon.
Data:            data/matches/*.json = our own match replays (only ours are exposed). analyze_matches.py parses
                 and replays them through the engine.
Open questions:  Level 4 per-match CPU limit is unpublished (Level 3: 15 s CPU / 20 s wall); keep models small.
