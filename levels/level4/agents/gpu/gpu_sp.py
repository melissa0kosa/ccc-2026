"""GPU-batched self-play: N games in one process, every leaf evaluation of every game in ONE torch forward.

Search = the referee's own colony_one.mcts internals (_descend/_set_priors/_backup/...), only `run`
is re-written as a generator that *yields* the planes it wants evaluated instead of calling the
evaluator. So semantics are identical to the server (PUCT 1.5, FPU 0, batch 8 + virtual loss,
2 actions per turn, sign flips only at a turn change, warmup noise/temperature).

Output records = agents/hand/gen_data.py format (sp_v4f.pkl):
    (planes[:13] uint8 (13,8,8), scalars x[13:,0,0] float32 (3,), z float (side to move), idx int16, cnt int16, q float)

Run (PyCharm: open this file, Run; defaults below):
    uv run python levels/level4/agents/gpu/gpu_sp.py [--games 512] [--par 256] [--model v2|rsv125] [--tag x]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
L4 = HERE.parents[1]
AZ = L4 / "agents" / "az"
for p in (str(L4), str(AZ)):
    if p not in sys.path:
        sys.path.insert(0, p)

from c1 import CYAN, ColonyOneEnv, encoding, mcts as M  # noqa: E402
from colony_one.rules import legal_actions_for_team  # noqa: E402

CKPT = {"v2": (AZ / "ckpt_res" / "az_res4f_v2.pt", 1.0),       # az_res4f_v2
        "rsv125": (AZ / "ckpt_res" / "az_res4f_v2.pt", 1.25)}  # deriv_4f2_rsv125 = v2 with dv x1.25


def load_torch_model(name="rsv125", device="cuda"):
    import torch
    import reslib
    path, vscale = CKPT[name] if name in CKPT else (Path(name), 1.0)
    net = reslib.ResFast(32, 2, kind="v4f")
    net.load_state_dict(torch.load(str(path), map_location="cpu")["net"])
    if vscale != 1.0:
        with torch.no_grad():
            net.vfc2.weight.mul_(vscale)
            net.vfc2.bias.mul_(vscale)
    return net.to(device).eval()


class TorchEval:
    """(B,16,8,8) float32 numpy -> ((B,4096) float32, (B,) float32), fp32 on GPU (TF32 off to match ONNX)."""

    def __init__(self, net, device="cuda", chunk=2048):
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch, self.net, self.device, self.chunk = torch, net, device, chunk
        self.calls = self.pos = 0
        self.t = 0.0

    def __call__(self, x):
        torch = self.torch
        t0 = time.perf_counter()
        outs_l, outs_v = [], []
        with torch.inference_mode():
            for i in range(0, len(x), self.chunk):
                xb = torch.from_numpy(np.ascontiguousarray(x[i:i + self.chunk])).to(self.device, non_blocking=True)
                lg, v = self.net(xb)
                outs_l.append(lg.float().cpu().numpy())
                outs_v.append(v.float().cpu().numpy())
        self.calls += 1
        self.pos += len(x)
        self.t += time.perf_counter() - t0
        return np.concatenate(outs_l), np.concatenate(outs_v)


# ------------------------------------------------------------------ search as a generator
def search_gen(board, team, actions_left, round_number, cfg, rng):
    """mcts.run, line for line, with `evaluate(planes)` replaced by `yield planes`."""
    root = M.Node(team, actions_left, round_number)
    in_warmup = M.move_number(team, actions_left, round_number) <= cfg.warmup_actions
    dirichlet = ((cfg.dirichlet_alpha, cfg.dirichlet_eps) if cfg.add_root_noise and in_warmup else None)
    legal = legal_actions_for_team(board, team)
    planes = encoding.encode_board(board, team, actions_left, round_number)
    logits_b, values = yield planes[None, ...]
    M._set_priors(root, legal, logits_b[0], float(values[0]), dirichlet, rng)

    batch_size = max(1, cfg.batch_size)
    vl = cfg.virtual_loss
    remaining = cfg.simulations
    while remaining > 0:
        pending = []
        done = 0
        try:
            for _ in range(min(batch_size, remaining)):
                path = []
                value, vsign, leaf = M._descend(root, board, +1, cfg.c_puct, vl, path)
                if leaf is not None:
                    M._apply_virtual_loss(leaf.path)
                    pending.append(leaf)
                    done += 1
                elif value is not None:
                    M._backup(path, value, vsign, release_virtual_loss=False)
                    done += 1
                else:
                    break
            if pending:
                logits_b, values = yield np.stack([p.planes for p in pending])
                for leaf, lg, v in zip(pending, logits_b, values, strict=True):
                    v = float(v)
                    M._set_priors(leaf.node, leaf.legal, lg, v)
                    M._backup(leaf.path, v, leaf.sign, release_virtual_loss=True)
        finally:
            for leaf in pending:
                leaf.node.pending = False
        if done == 0:
            break
        remaining -= done
    return root


def game_gen(cfg, rng):
    """One self-play game; yields eval requests; returns (records, cyan_won, n_actions, actions)."""
    env = ColonyOneEnv()
    recs, acts = [], []
    while not env.terminated:
        team, al, rnd = env.current_team, env.actions_left, env.round_number
        x = encoding.encode_board(env.board, team, al, rnd)
        root = yield from search_gen(env.board, team, al, rnd, cfg, rng)
        idx, cnt = M.visit_distribution(root)
        temp = M.warmup_temperature(M.move_number(team, al, rnd), cfg.warmup_temperatures, cfg.warmup_step_actions)
        ai = M.select_action_index(root, temp, rng=rng)
        nsum = sum(root.N.values())
        q = sum(root.W.values()) / nsum if nsum else 0.0
        recs.append((x[:13].astype(np.uint8), x[13:, 0, 0].copy(), team,
                     np.array(idx, np.int16), np.array(cnt, np.int16), float(q)))
        acts.append(int(ai))
        env.step(encoding.decode_action(ai, team, env.board))
    w = env.winner
    out = [(pl, sc, 1.0 if t == w else -1.0, i, c, q) for pl, sc, t, i, c, q in recs]
    return out, (w == CYAN), len(recs), acts


def play_many(evaluate, n_games, n_par, cfg, seed, log_every=30.0, on_progress=None):
    """Keep n_par games in flight; each round, one batched evaluate() over all pending requests."""
    ss = np.random.SeedSequence(seed)
    started = 0
    live = {}   # gid -> (gen, request)
    results = []

    def start():
        nonlocal started
        gid = started
        started += 1
        g = game_gen(cfg, np.random.default_rng(ss.spawn(1)[0]))
        live[gid] = [g, next(g)]

    for _ in range(min(n_par, n_games)):
        start()
    t0 = tlast = time.time()
    while live:
        gids = list(live)
        reqs = [live[g][1] for g in gids]
        sizes = [len(r) for r in reqs]
        lg, v = evaluate(np.concatenate(reqs))
        off = 0
        for gid, n in zip(gids, sizes):
            gen = live[gid][0]
            try:
                live[gid][1] = gen.send((lg[off:off + n], v[off:off + n]))
            except StopIteration as e:
                results.append(e.value)
                del live[gid]
                if started < n_games:
                    start()
            off += n
        if time.time() - tlast > log_every:
            tlast = time.time()
            el = tlast - t0
            print(f"  {len(results)}/{n_games} games, {sum(r[2] for r in results)} pos, {el:.0f}s, "
                  f"{60 * len(results) / el:.0f} games/min", flush=True)
            if on_progress is not None:
                on_progress(results)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--par", type=int, default=64)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--model", default="rsv125")
    ap.add_argument("--tag", default="test")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--warm", default="1.0,1.0,0.7,0.5", help="warmup temperatures, 4 actions each (gen_data.py default)")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    seed = a.seed if a.seed is not None else int(time.time() * 1000) % (2 ** 31)
    cfg = M.MCTSConfig(simulations=a.sims, warmup_temperatures=tuple(float(t) for t in a.warm.split(",")))
    ev = TorchEval(load_torch_model(a.model))
    t0 = time.time()
    out = HERE / "data" / f"gsp_{a.model}_{a.tag}.pkl"

    def save(res, dt):
        with open(out, "wb") as f:
            pickle.dump([r for g in res for r in g[0]], f)
        with open(out.with_suffix(".meta.pkl"), "wb") as f:
            pickle.dump(dict(actions=[g[3] for g in res], cyan_won=[g[1] for g in res], n=[g[2] for g in res],
                             seed=seed, model=a.model, sims=a.sims, warm=a.warm, secs=dt), f)

    res = play_many(ev, a.games, a.par, cfg, seed, on_progress=lambda r: save(r, time.time() - t0))
    dt = time.time() - t0
    data = [r for g in res for r in g[0]]
    npos = len(data)
    print(f"[{a.tag}] {len(res)} games, {npos} positions, {dt:.1f}s -> {60 * len(res) / dt:.0f} games/min, "
          f"{60 * npos / dt:.0f} pos/min | cyan wins {np.mean([r[1] for r in res]):.2f}, "
          f"avg len {np.mean([r[2] for r in res]):.1f} | gpu {ev.t:.1f}s in {ev.calls} calls "
          f"({ev.pos / max(1, ev.calls):.0f} pos/call) | peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB",
          flush=True)
    save(res, dt)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
