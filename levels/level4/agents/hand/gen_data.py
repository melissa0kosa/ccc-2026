"""Self-play data: positions + game outcome (side-to-move view) + root visit counts.
usage: gen_data.py A.onnx B.onnx games sims out.pkl [workers]   (B may equal A)"""
import pickle, sys, time, secrets
from multiprocessing import get_context
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np


def play(args):
    a, b, g, sims, seed = args
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from c1 import evaluator, ColonyOneEnv, mcts, encoding, CYAN
    rng = np.random.default_rng(seed + g)
    cfg = mcts.MCTSConfig(simulations=sims, warmup_temperatures=(1.0, 1.0, 0.7, 0.5))
    evs = {}
    for s in (a, b):
        evs[s] = evaluator(str(Path(s).resolve()))
    cyan_spec, mag_spec = (a, b) if g % 2 == 0 else (b, a)
    env = ColonyOneEnv()
    recs = []
    while not env.terminated:
        team = env.current_team
        ev = evs[cyan_spec if team == CYAN else mag_spec]
        x = encoding.encode_board(env.board, team, env.actions_left, env.round_number)
        root = mcts.run(env.board, team, env.actions_left, env.round_number, ev, cfg, rng=rng)
        idx, cnt = mcts.visit_distribution(root)
        temp = mcts.warmup_temperature(mcts.move_number(team, env.actions_left, env.round_number),
                                       cfg.warmup_temperatures, cfg.warmup_step_actions)
        ai = mcts.select_action_index(root, temp, rng=rng)
        nsum = sum(root.N.values())
        q = sum(root.W.values()) / nsum if nsum else 0.0
        recs.append((x[:13].astype(np.uint8), x[13:, 0, 0].copy(), team,
                     np.array(idx, np.int16), np.array(cnt, np.int16), float(q)))
        env.step(encoding.decode_action(ai, team, env.board))
    w = env.winner
    out = []
    for planes, sc, team, idx, cnt, q in recs:
        out.append((planes, sc, 1.0 if team == w else -1.0, idx, cnt, q))
    return out, (w == CYAN), len(recs)


if __name__ == "__main__":
    a, b, games, sims, outp = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    seed = secrets.randbits(30)
    t = time.time()
    with get_context("spawn").Pool(workers) as pool:
        res = pool.map(play, [(a, b, g, sims, seed) for g in range(games)], chunksize=2)
    data = [r for game in res for r in game[0]]
    print(f"{games} games, {len(data)} positions, cyan wins {np.mean([r[1] for r in res]):.2f}, "
          f"avg len {np.mean([r[2] for r in res]):.1f}, {time.time()-t:.0f}s")
    with open(outp, "wb") as f:
        pickle.dump(data, f)
