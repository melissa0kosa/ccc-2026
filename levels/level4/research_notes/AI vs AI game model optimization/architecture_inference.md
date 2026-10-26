# Architecture and CPU-inference optimization for a tiny COLONY: ONE policy/value ONNX model

Scope: network shape, policy-head choice, latency budget, ONNX Runtime (ORT) CPU behaviour, exact hand-feature
layers inside the graph, distillation, export pitfalls. All latency numbers below were **measured in this session**
on the team machine with ORT `intra_op_num_threads=1, inter_op_num_threads=1` (the referee's setting), default
graph optimization level (= referee default), batch 8, sparse random 0/1 boards, 300 timed calls after 30 warm-up.
The box was shared with three live training jobs, so expect about ±25% noise (res4x48 measured 2.87 ms in one run
and 3.65 ms in another; res6x64 measured 9.5 and 8.5 ms). Benchmark scripts:
`C:\Users\melis\AppData\Local\Temp\claude\C--Users-melis-Downloads-cloudflight-coding-contest\2537b5b1-8fcc-4ac8-8fd3-b627f7b7b210\scratchpad\bench_arch.py`, `bench2.py`, `bench3.py`
(scratchpad; copy them into the level folder if you want to keep them).

## Q0. Hard facts from the local referee code (read-only)

### Takeaway
The referee runs every model **single-threaded** on CPU with ORT defaults (all graph optimizations including NCHWc on).
INT8 is impossible: no Quantize/Dequantize ops are allowed. FP16 is usable only as *storage* via `Cast`. Each action
costs 1 batch-1 call plus 13 calls at batch ≤8.

### Cited Findings
- The referee's `OnnxEvaluator` sets `intra_op_num_threads = num_threads` (default 1, "The referee pins this so that every submission gets the same compute") and `inter_op_num_threads = 1`, `providers=["CPUExecutionProvider"]`. It does not set `graph_optimization_level`, so ORT's default applies. The upload validator also uses 1/1 threads. — [onnx_runner.py L206-243](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/onnx_runner.py); [submission_policy.py L651-656](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/submission_policy.py)
- Allow-list (complete): Add, Sub, Mul, Div, Pow, Neg, Abs, Sqrt, Exp, Log, Min, Max, Sum, Mean, Clip, Equal, Greater, GreaterOrEqual, Less, LessOrEqual, And, Or, Not, Xor, Where, Relu, LeakyRelu, PRelu, Elu, Selu, Sigmoid, Tanh, Softplus, Softsign, HardSigmoid, Softmax, LogSoftmax, Conv, Gemm, MatMul, BatchNormalization, InstanceNormalization, LayerNormalization, AveragePool, MaxPool, GlobalAveragePool, GlobalMaxPool, Identity, Cast, Flatten, Transpose, Concat, Reshape, Squeeze, Unsqueeze, Split, ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd, Gather, Slice. **`Constant` is not on it.** Neither are QuantizeLinear/DequantizeLinear/QLinearConv/ConvInteger/MatMulInteger, nor Pad/Expand/Tile/OneHot/Scatter*/Range/ConstantOfShape. — [submission_policy.py L109-152](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/submission_policy.py)
- Shape and index inputs must be constant initializers: Reshape(1), Squeeze(1), Unsqueeze(1), Slice(1-4), Split(1), Gather(1). Cast may target only FLOAT, FLOAT16, INT32, INT64, BOOL. — [submission_policy.py L155-177](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/submission_policy.py)
- Structural budgets: MAX_NODES 1024, MAX_INITIALIZERS 512, MAX_TENSORS 1024, MAX_ATTRIBUTES 4096, per-tensor decoded ≤16 MiB, total decoded ≤32 MiB, file ≤5 MiB, opset 13-18. Check limits: 2 s CPU / 3 s wall. Level 3 match: 15 s CPU, 20 s wall, 512 MiB. — [submission_policy.py L42-67](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/submission_policy.py)
- The local validator's match check plays one Level 3 match (referee MCTS, 60 sims) **against GreedyAgent** and stops at 13.5 s CPU (the server's RLIMIT_CPU is 15 s). It does not test 100 sims against another network. — [tools/validate_submission.py L28-29, L75-79, L295-309](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/tools/validate_submission.py)
- MCTS call pattern: one root evaluation with `planes[None, ...]` (batch 1), then batches of `min(batch_size=8, remaining)` leaves until `simulations` are spent. 100 sims gives 12×b8 + 1×b4 + 1×b1 = **14 calls per action**. — [mcts.py L326, L599-635](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/mcts.py)
- Action geometry: the action index is from_square×64 + to_square, with no move/shot kind axis. The real branching factor is 12-30. Move ranges: Ping 2 (diag capture 2), Daemon 3, Trace move 1 / shoot 3, Ice 2 in 8 directions, Kernel 1, Exploit 8 knight jumps. **Every legal (from,to) pair is one of 32 relative offsets**: 8 directions × distance 1-3 (24) plus 8 knight jumps. — [encoding.py L54-66](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/encoding.py); [pieces.py L58-65](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/pieces.py); [rules.py L36-39](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/rules.py)

### Inferences
- With one intra-op thread there is no thread pool, so ORT spinning and thread settings are irrelevant. Latency is single-core compute plus per-node dispatch overhead.

### Gaps
- The Level 4 per-match CPU limit, and whether both seats' models run in one rlimited process (shared 15 s) or in separate processes, is not published locally.

## Q1. Which architectures work best at tiny size, and what are their CPU latencies at batch 8?

### Takeaway
On one core, ORT runs about **25-30 GMAC/s**. Latency at batch 8 ≈ **0.3 ms per million MACs per position**: a 3×3 conv
costs 9·C²·64 MACs per position (C=32: 0.59M, C=48: 1.33M, C=64: 2.36M). Within a ≤3 ms/call budget the sweet spot
is **res4x48, res6x32, res8x32-SE, or exact ray features + res4x32**. A 6x64 net costs 8.5-9.5 ms and is too expensive.
Global-pooling bias is free (it even saves time). SE costs about 45 µs per block at batch 8.

### Cited Findings
- Measured single-thread ORT latency (ms per call; b8 = batch 8, b1 = batch 1; all with a conv-1x1 64-plane policy head and a conv1x1(4)->FC64->tanh value head unless noted). Source: local benchmark, `bench_arch.py`/`bench2.py`/`bench3.py` (scratchpad, see header):

| model | params | file KB | nodes | b8 ms | b1 ms | own CPU / 80-action game* |
|---|---|---|---|---|---|---|
| hand_v1 (team, fixed features->linear) | 5.8k | 69 | 383 | 1.04 | 0.49 | 0.56 s |
| res4x32 | 97k | 385 | 33 | 1.63-1.84 | 0.25-0.48 | 0.9 s |
| res4x32 + bilinear head d=16 | 96k | 382 | 38 | 1.36 | 0.26 | 0.7 s |
| res6x32 | 134k | 531 | 43 | 1.89 | 0.26 | 1.0 s |
| ray features (hand.py style) only | 42k | 169 | 26 | 1.27 | 0.15 | 0.67 s |
| **hybrid: ray features + res4x32** | 116k | 461 | 46 | **2.23** | 0.48 | 1.2 s |
| dw8x64 (depthwise-separable) | 106k | 426 | 69 | 2.75 | 0.36 | 1.4 s |
| **res8x32 + SE** | 197k | 800 | 157 | **2.87** | 0.45 | 1.5 s |
| **res4x48** | 193k | 759 | 33 | **2.87-3.65** | 0.36-0.47 | 1.5-1.9 s |
| res4x48 + KataGo gpool bias (2 blocks) | 185k | 730 | 53 | 2.89 | 0.38 | 1.5 s |
| res4x48, conv64 head w/o Transpose | 193k | 759 | 32 | 3.32 (vs 3.65 same run) | 0.46 | 1.7 s |
| res4x48 + relative-offset gather head (3x3, K=32) | 208k | 833 | 33 | 4.16 (vs 3.65 same run) | 0.53 | 2.2 s |
| res3x64 | 252k | 987 | 28 | 3.98 | 0.48 | 2.1 s |
| hybrid: ray features + res4x48 | 217k | 853 | 46 | 4.59 | 0.59 | 2.4 s |
| dw8x96 | 200k | 793 | 69 | 5.03 | 0.78 | 2.6 s |
| res6x48 | 276k | 1085 | 43 | 5.13 | 0.73 | 2.7 s |
| res6x64 | 473k | 1855 | 43 | 8.5-9.5 | 1.1-1.25 | 4.5-5.0 s |
| res6x64 + gpool bias (2 blocks) | 458k | 1800 | 63 | 9.03 | 0.95 | 4.7 s |
| res6x64 + SE | 511k | 2021 | 121 | 9.63 | 1.82 | 5.1 s |
| res6x64 + bilinear head d=32 | 473k | 1856 | 48 | 14.85 | 1.41 | 7.8 s |
| res6x64 + FC policy head (2 planes->4096) | 998k | 3904 | 44 | 12.58 | 3.17 | 6.7 s |
| res8x64 + SE | 672k | 2654 | 157 | 19.2 | 2.34 | 10.1 s |
| res6x96 | 1.03M | 4044 | 43 | 53 (cpu 24.7) | 6.6 | >13 s |
| res10x128 | 3.0M | 11711 (too big) | 63 | 61-83 | 7.2 | >30 s |
| MLP 1024-256-256 + FC 4096 | 1.38M | 5396 (>5 MiB fp32) | 9 | 0.67 | 0.24 | 0.36 s |
| MLP 512 hidden, fp16 storage | 2.89M | 5647 (too big even fp16) | 13 | 2.15 | 0.67 | - |

  *Own CPU per game = 40 own actions × (13·b8 + 1·b1) on this machine, model time only (excludes MCTS Python overhead).
- Lc0 SE block: residual block of two 3x3 convs + SE (global average pool -> FC to SE_CHANNELS ≈32 -> ReLU -> FC to 2×FILTERS split into W and B; out = sigmoid(W)·x + B) + skip + ReLU. BN is folded into conv weights. Typical sizes are 10x128, 20x256, 24x320. — [Lc0 NN backend docs](https://lczero.org/dev/backend/nn/)
- KataGo global pooling bias: take channels G, apply BN+ReLU, pool to mean, mean×board-width-scale and max (3c values), FC to cX outputs, and add as per-channel bias to channels X. It follows the first conv of 2-3 residual blocks and the first conv of the policy head. The paper says it "greatly improves the later stages of training" and that "global context should help even in domains without explicit nonlocal interactions". — [KataGo paper, arXiv 1902.10565 §3.3, App. A.2](https://arxiv.org/abs/1902.10565)
- KataGo found standard bottleneck blocks "substantially worse than a normal residual block at equal total compute". Its nested bottleneck (four 3x3 convs between 1x1s) was a win only at large scale (b18c384). — [KataGo Methods doc](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
- Lc0 distilled CPU nets: Kappe distilled ID11258 into SE nets from 16x2-se to 128x10-se to find the best dimensions for CPU. He reported 24x3-se as ~300 Elo stronger than 16x2-se. On slow CPUs the "sweet spot" depends on nodes per second. — [dkappe Distilled Networks wiki](https://github.com/dkappe/leela-chess-weights/wiki/Distilled-Networks); [TalkChess thread](https://talkchess.com/viewtopic.php?p=789704)

### Inferences
- Because the simulation count is fixed, a smaller and faster net buys **no** extra search. Use the whole safe budget (Q3) on evaluation quality, and stop there.
- Receptive field: a corner-to-corner Chebyshev distance of 7 needs ≥7 stacked 3x3 convs. res4 gives 9 convs (stem + 8), enough. res3 gives 7, borderline. One or two global-pooling-bias (or SE) layers give board-wide context at almost no cost (measured: gpool on res4x48 cost +0.02 ms and reduced params, because the pooled channels skip the second conv).
- Depth beats width at this size: res8x32-SE (197k) and res4x48 (193k) cost the same. Kappe's Lc0 sweep suggests small SE towers are a good CPU choice. Train both if time allows; otherwise res4x48 + 2 gpool blocks is the simplest strong default.
- Depthwise-separable convs are not a speed win here: dw8x64 is about as fast as res4x48 with half the params, because grouped convs run less efficiently in ORT.
- Long-range exact tactics (sliding rays with blocking up to distance 3, knight-like jumps, shots) are hard for a 100-200k-param convnet to learn exactly. The hybrid (exact ray planes concatenated with the input) costs only about +0.5-1.3 ms and is the best use of the budget. See Q5.
- A pure MLP is very cheap (GEMM-bound) but the 4096-wide FC policy layer eats the file budget. MLP-256 needs fp16 storage to fit, and MLP-512 doesn't fit even in fp16. Only consider an MLP if it sits on compact hand features (tens to hundreds of scalars) with a low-rank policy (e.g. hand prior + small correction).

### Gaps
- No published CPU-ORT latency tables for small Othello or Connect-4 nets were found. The table above is the only latency evidence, from one shared machine.
- The server CPU type and speed are unknown. ORT's NCHWc kernels depend on AVX2 vs AVX-512.

## Q2. Policy head for the 4096 from-to action space

### Takeaway
Use a **1x1 conv to 64 planes, with channel = from-square and spatial = to-square, so a plain Flatten already
gives from·64+to and no Transpose is needed** (64·C params, fastest). For a better inductive bias at the same cost, use the
Lc0-style **relative-offset head**: a 1x1 conv to K=32 offset planes at the from-square, then one `Gather` with a constant
4096-index initializer. The FC head doubles params and adds about 3 ms. The bilinear/attention head gives no speed gain
for a convnet and was slower at C=64.

### Cited Findings
- Lc0 convolutional policy: conv FILTERS->FILTERS, conv -> 80×8×8 (73 used), then a fixed-mapping gather to 1858 moves. Classical head: conv -> POLICY_CONV_SIZE planes -> FC -> 1858. — [Lc0 NN backend docs](https://lczero.org/dev/backend/nn/)
- Chessformer / Lc0 attention policy: linear projections of the 64 square tokens to query vectors (start square) and key vectors (destination square), "both with depth equal to the depth of the encoder body". Scaled dot-product gives a 64×64 logit matrix. The paper calls the from-to formulation "without sacrificing performance". — [Chessformer, arXiv 2605.19091 §3.3](https://arxiv.org/pdf/2605.19091)
- Measured on res6x64: conv64 head 8.5-9.5 ms, FC head 12.6 ms (+525k params, file 1.9 -> 3.9 MB), bilinear d=32 14.9 ms. On res4x32 the bilinear head with d=16 was 1.36 ms vs 1.63-1.84 ms for conv64, roughly a tie. On res4x48 in the same run: conv64 with Transpose 3.65 ms, without Transpose 3.32 ms, relative-offset head with a **3x3** conv 4.16 ms. — local benchmark (scratchpad `bench_arch.py`, `bench3.py`)
- The `Gather` index input must be constant. An initializer from `register_buffer` qualifies: the exported relative-offset model's ops were Add, Conv, Flatten, Gather, Gemm, Relu, Reshape, Tanh. — [submission_policy.py L155-162](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/submission_policy.py); local benchmark `bench3.py`
- The referee masks illegal actions itself, so logits at unreachable (from,to) indices never matter. — [submission.md "Output mappings"](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/docs/submission.md)

### Inferences
- Parameter cost per head at C channels: conv64 = 64C+64 (3.1k at C=48). Relative-offset 1x1 = 32C+32 (1.6k), 3x3 = 288C (14k). FC from 2 planes = 128·4096 = 524k (2 MB fp32). Bilinear = 2dC.
- Relative-offset head recipe (index table verified in `bench3.py`): `OFFS` = 8 dirs × k∈{1,2,3} + 8 knight jumps. For from f and offset j with target t inside the board, `idx[f*64+t] = j*64+f`. Unreachable pairs point at index 0. Then `pol = index_select(conv1x1(h).flatten(1), 1, idx_buffer)`. Use a 1x1 conv; the 3x3 cost +0.5-0.8 ms with no clear benefit. Because the observation is canonicalised (180° rotation for Magenta, `encoding._rotate`), the offset table is perspective-invariant as long as the policy index uses the same canonical frame. **Verify the frame against `encoding.legal_mask`/the team's training data before trusting it.**
- Optional: KataGo-style policy gpool bias (pooled trunk -> FC -> bias on the policy conv's input) adds global context to move priors for about 20 µs.

### Gaps
- No ablation was found that compares conv-absolute, relative-offset and bilinear heads on small convnets. Strength differences must be tested in the team arena.

## Q3. How much does speed matter, and what is the safe latency budget per call?

### Takeaway
Speed buys nothing beyond staying under the CPU limit, because sims are fixed at 100. Per own action: 13·t8 + t1. Take
the worst case: an 80-action game (40 own), a 15 s rlimit possibly shared by both seats and the MCTS Python overhead,
and a server 1.5-2× slower than the dev box. Then aim for **≤3 ms per batch-8 call locally (hard ceiling about 5 ms)**,
i.e. ≤1.5-2.5 s own model CPU per game on this machine.

### Cited Findings
- 14 calls per action at 100 sims (12×b8 + 1×b4 + 1×b1). — [mcts.py L326, L599-635](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/mcts.py)
- Level 3 match RLIMIT_CPU is 15 s and covers the whole match process. The validator plays against GreedyAgent and stops at 13.5 s. "Matches differ in length and machines differ in speed, so keep clear of the limits rather than just under them." — [submission.md "Limits"](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/docs/submission.md); [validate_submission.py L75-79](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/tools/validate_submission.py)
- Team measurement: hand_v1 games take about 1.0 s CPU per game for both sides together. — [agents/hand/RESULTS.md](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/hand/RESULTS.md)

### Inferences
- Budget formula: own_cpu = A_own × (13·t8 + t1) × server_factor. Worst case, shared limit: 15 s × 0.7 safety ≈ 10.5 s. Subtract about 1.5 s for interpreter, numpy, ORT import, session init and MCTS Python, leaving about 9 s. Split by 2 seats gives about 4.5 s per model. Divide by (40 actions × 13.5 call-equivalents) to get about 8 ms per call on the server, and divide by a server factor of 1.5-2 to get **about 4-5.5 ms locally**. Adding margin for longer games and an unknown opponent gives **≤3 ms locally as the target**.
- If the limit turns out to be per seat (separate processes), the budget about doubles (≤6-8 ms locally) and res6x48 or a ray + res6x48 hybrid becomes affordable. If "35-80 actions" means per side rather than per game, halve every budget.
- Per-game cost scales linearly with game length, so a model that is fine on average can time out in the longest games. Budget for the longest game seen in the team arena logs, not the mean.

### Gaps
- MCTS Python overhead per simulation was not measured, because running the env code was off-limits here. Measure it with `models/random.onnx` (near-zero model cost) in the team arena: CPU/game with random.onnx is the overhead floor.
- The server-to-dev CPU speed ratio is unknown. Re-check once the server reports any CPU time, e.g. the validator's match report.

## Q4. ONNX Runtime CPU performance: fusion, layout, FP16/INT8, threads

### Takeaway
The referee keeps ORT's default (all optimizations), so Conv+BN, Conv+ReLU and the NCHWc layout are applied automatically. That
is worth **3.2×** on a ResNet (30 -> 9.5 ms). Your job is to keep the graph "conv-shaped": few nodes, no Transposes or Gathers inside
the trunk, channels a multiple of 8/16. INT8 is not allowed. FP16 helps only as storage (halves the file at no latency cost).

### Cited Findings
- ORT levels: Basic = constant folding, redundant-node elimination, Conv+Add/Conv+Mul/Conv+BN fusion. Extended (CPU) = GEMM+Activation, MatMul+Add, Conv+Activation. Layout (`ORT_ENABLE_ALL`) = NCHWc layout optimizer, CPU EP only. "All optimizations are enabled by default." Layout-optimized saved models need the same instruction set (e.g. AVX2). — [ORT graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
- Measured res6x64 at batch 8, single thread: no optimizations 30.0 ms, Extended (fusions, no NCHWc) 19.8 ms, All/default 9.5 ms. — local benchmark `bench_arch.py`
- The PyTorch exporter (eval mode, `do_constant_folding=True`, opset 17, `dynamo=False`) already folded BN into Conv: the exported res6x64 graph contained only Add, Conv, Flatten, Gemm, Relu, Reshape, Tanh, Transpose, with **no BatchNormalization nodes**. — local benchmark `bench_arch.py`
- Threading: intra-op threads parallelize inside operators. Spinning is on by default and "consumes more CPU cycles". The inter-op pool exists only in `ORT_PARALLEL` mode. — [ORT thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html). The referee pins 1 intra-op and 1 inter-op thread — [onnx_runner.py L232-234](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/onnx_runner.py)
- FP16 storage test: converting initializers ≥4096 elements to float16 plus a `Cast(to=FLOAT)` cut res8x96-SE from 5661 KB to 2904 KB and res10x128 from 11711 KB to 5868 KB. b8 latency was unchanged within noise (61 vs 73 ms; 51 vs 83 ms under contention) and session init stayed at 15-31 ms CPU. — local benchmark `bench_arch.py`
- No QuantizeLinear/DequantizeLinear/QLinear*/ *Integer ops are on the allow-list. — [submission_policy.py L109-139](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/submission_policy.py)
- Per-node overhead: hand_v1 has 383 nodes and only 5.8k params, yet costs 1.04 ms at batch 8. SE adds about 13 nodes per block. res8x32-SE (157 nodes) took 2.87 ms vs about 2.5 ms extrapolated from res6x32 without SE, i.e. about 45 µs per SE layer. — local benchmark `bench2.py`

### Inferences
- Rule of thumb for this box: at batch 8, ≈0.3 ms per 1M MAC per position, plus about 2-3 µs per node.
- Keep channel counts at multiples of 16, so NCHWc blocks of 8 (AVX2) or 16 (AVX-512) tile exactly. This is ORT-internals knowledge, not verified here.
- Put Transpose/Reshape/Gather only in the heads, never between trunk convs: each one forces an NCHWc reorder. The policy conv with channel = from-square needs no Transpose at all (saved about 0.3 ms on res4x48).
- **Do not submit an ORT-saved optimized model** (`optimized_model_filepath`). Fused and NCHWc nodes live in Microsoft domains (`com.microsoft`, `com.microsoft.nchwc`), which the domain allow-list (`""`, `"ai.onnx"`) rejects, and the referee re-optimizes at load anyway. This is inferred from ORT internals and not tested here.
- FP16 storage matters only if file size binds: an FC policy head, an MLP, or a convnet above about 1.3M params. In the ≤3 ms budget, convnets are ≤300k params (≤1.2 MB fp32), so stay fp32 and avoid the fp16 rounding mismatch with training. If you do use it, keep small tensors (biases, index tables) fp32. INT64 Gather indices must stay INT64.
- Thread count, spinning and execution mode cannot be tuned: the referee fixes the session options.

### Gaps
- FP16 *compute* on ORT CPU was not benchmarked. With a Cast to float right after the initializer, compute stays fp32, which is the only reliable path.

## Q5. Exact rule-based features with allowed ops, and fusing them with learned layers

### Takeaway
Ray propagation with blocking works as **a grouped 3x3 "shift" conv applied to `r_k × empty` k times**, the hand.py
pattern (0.15 ms at b1, 1.27 ms at b8 for 80 ray channels × 3 steps + a 1x1 combine). Feed the resulting exact threat planes
into the trunk by **concatenating them with the 16 input planes before the stem conv**. Optionally also feed hand scalar
features into the value FC, initialised so the network starts as hand_v1. Every fixed tensor goes in via `register_buffer`
(it becomes an initializer). Any leftover `Constant` nodes are converted by `c1.fix_constants`.

### Cited Findings
- Team pattern: `x0 = conv1x1(x, w_sel)` selects 80 ray channels (2 sides × 4 sliders × 8 dirs + kernel rays). Then `r1 = conv(x0, w_shift, padding=1, groups=80)`, `r2 = conv(r1*empty, ...)`, `r3 = conv(r2*empty, ...)`, then `conv1x1(cat[r1,r2,r3], w_comb)` gives 24 attack/move maps. Knight-type jumps use one 5x5 conv (`w_jk`, padding 2), adjacency uses a 3x3 conv, and the ring/firewall uses `amax` over masks. — [agents/hand/hand.py L51-137, L181-227](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/hand/hand.py)
- Measured: this ray extractor (80 ray channels, 3 steps, combine to 32 planes, clamp) costs 1.27 ms at b8 and 0.15 ms at b1 on its own. Concatenated into a res4x32 stem it costs 2.23 ms total (vs 1.63-1.84 ms for plain res4x32), and into res4x48 4.59 ms (vs 2.87-3.65 ms). Exported ops were only allowed ones (Conv, Mul, Sub, ReduceSum, Concat, Clip, ...), and the buffers became initializers. — local benchmark `bench2.py`
- Shape and index arguments must be initializers. The team's `fix_constants` rewrites every exporter-emitted `Constant` node into an initializer and pins the I/O dims. — [c1.py L48-90](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/c1.py); [submission_policy.py L155-162](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/submission_policy.py)

### Inferences
- Allowed-op building blocks for exact features:
  - shift by (dr,dc): a fixed one-hot 3x3 (or 5x5) Conv with `padding` set as a Conv *attribute*. The `Pad` op is banned, but Conv padding is fine.
  - blocking: multiply by the `empty` plane (Sub from 1 of a ReduceSum over occupancy) before each further shift. The ray to distance k is k chained shifts. Equivalently, a "cumulative product along the ray" is just this chain; there is no CumProd op (not allowed and not needed).
  - union of attacks: Max, or Clip(sum, 0, 1). Counts: ReduceSum. Global flags: ReduceMax / GlobalMaxPool.
  - piece selection, combining planes: a 1x1 Conv with fixed weights is cheaper than many Slice/Mul nodes, and keeps the node count low (hand_v1's 383 nodes are already 1/3 of MAX_NODES = 1024).
  - board-region masks (rings, ranks): fixed initializer planes broadcast-multiplied.
- Group all directions and pieces into one grouped conv per step (as hand.py does) rather than one conv per direction. Node count, not FLOPs, dominates these small ops.
- Fusion options, cheapest to strongest:
  1. **Input-plane concat** (recommended): `stem(cat[x, hand_planes])`. Costs about +0.4-1.3 ms and the trunk gets exact tactics from layer 1.
  2. **Value residual on hand_v1**: `v = tanh(w_hand·F_hand + b + MLP(trunk_pool))`, with w_hand initialised to hand_v1's fitted weights and the MLP's last layer zero-initialised. Training starts at hand_v1 strength and can only add.
  3. **Policy prior residual**: `logits = hand_prior_logits + conv_policy(h)`, with the conv head zero-initialised. Same "start at the teacher" trick.
- Mark the hand-extractor buffers non-trainable (buffers are not parameters by default), so training cannot break the exact features.

### Gaps
- Whether hybrid input planes beat a plain res6x32 of equal cost on this game is untested. Only the latency was measured.

## Q6. Knowledge distillation into a small student

### Takeaway
Train the student on **legal-masked KL(policy target ‖ student) + MSE(value)**. Prefer **MCTS visit distributions and game
outcomes** (teacher + search) over raw teacher logits as targets. With raw logits use T≈1-2 and scale the soft loss by T².
Students can match or beat their teachers, but in this setting the realistic gain is "teacher + 100-sim search distilled into the
student", i.e. one policy-improvement step.

### Cited Findings
- Hinton et al. distil by matching temperature-softened softmax outputs ("compress the knowledge in an ensemble into a single model"). The paper also recommends multiplying the soft-target loss by T², because soft-target gradients scale as 1/T². — [Hinton, Vinyals, Dean 2015, arXiv 1503.02531](https://arxiv.org/abs/1503.02531) (the T² detail is from the paper body, not re-verified in this session)
- Policy distillation (Rusu et al.): the distilled student is "dramatically smaller and more efficient" and "performs at the expert level". A multi-task distilled agent "outperforms the single-task teachers". — [Rusu et al. 2015, arXiv 1511.06295](https://arxiv.org/abs/1511.06295)
- Born-Again Networks: students with the same architecture as their teachers "outperform their teachers significantly". — [Furlanello et al. 2018, arXiv 1805.04770](https://arxiv.org/abs/1805.04770)
- Lc0 practice: large nets distilled into 16x2 to 128x10 SE students for CPU play. — [dkappe Distilled Networks wiki](https://github.com/dkappe/leela-chess-weights/wiki/Distilled-Networks)

### Inferences
- Recipe for this team:
  - Policy targets: visit counts from referee MCTS (100 sims, same `MCTSConfig`) with the best current model, masked to legal moves.
  - Value targets: a mix of game outcome z and the teacher/search root value (e.g. 0.5/0.5) to cut variance.
  - Loss = CE(π_target, softmax(masked student logits)) + c_v·MSE(v, target), c_v ≈ 1-2. Use AdamW with a cosine LR.
  - Augmentation: there is no board symmetry unless the rules are mirror-symmetric (check the left-right mirror before using it).
- Data scale: a 100-200k-param student needs roughly 10⁵-10⁶ deduplicated positions to fit without overfitting. **This is a heuristic, not sourced.** Watch held-out policy KL and value MSE, and stop once the arena win-rate plateaus.
- Can the student beat the teacher inside MCTS? Yes if the targets come from teacher+search (AlphaZero/Expert-Iteration style policy improvement), or if the student is larger, or if it has the hand features the teacher lacks. When distilling raw logits into a smaller student, expect parity at best. Evaluate only in the arena at 100 sims, since policy KL does not predict MCTS strength well.
- Initialise from hand_v1 (Q5 fusion option 2/3), so even a briefly trained student is not weaker than the deployed model.

### Gaps
- No source was found that quantifies data requirements or student-vs-teacher Elo for distillation into tiny nets used under small fixed-sim MCTS. AlphaZero/ExIt (arXiv 1712.01815, 1705.08439) support the "search targets improve the policy" claim, but they were not re-fetched in this session.

## Q7. Export pitfalls: dynamic axes, opset, Constant nodes, checker, onnxsim

### Takeaway
Use `torch.onnx.export(..., opset_version=17, dynamo=False, do_constant_folding=True, external_data=False,
dynamic_axes={observation/policy/value: {0: "batch"}})`, then `fix_constants`, pin the policy dim to 4096, and run
`onnx.checker` plus the local validator on **every** export. Write shapes as literals: `flatten(1)`, `reshape(-1, 4096)`, `reshape(-1)`.

### Cited Findings
- Traps: external weights (PyTorch 2.9+ turns `external_data` on by default) and a frozen batch axis. Value must be (B,), not (B,1). Export in eval mode. The validator checks batch 1 and 16 on a real-position corpus. — [submission.md](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/docs/submission.md)
- Team exporter: opset 17, `dynamo=False`, `do_constant_folding=True`, `external_data=False`, dynamic batch on all three tensors, then `fix_constants`. The rules: literal ints only for reshapes, never `x.view(x.size(0), ...)` ("makes a data-dependent shape"), and end the value with tanh. `export_util` pins `policy` dim 1 to 4096 and retries the validator on "too slow". — [c1.py L100-125](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/c1.py); [agents/distill/export_util.py](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/agents/distill/export_util.py)
- The validator runs `onnx.checker.check_model(bytes)` before the contract checks. — [submission_policy.py L640-645](file:///C:/Users/melis/Downloads/cloudflight-coding-contest/levels/level4/data/colony-one-env/colony_one/submission_policy.py)
- All 20+ benchmark models exported with this recipe (incl. Gather with buffer indices, SE `[:, :c]` slices, gpool Concat/ReduceMax, bilinear MatMul/Transpose, fp16 Cast) used only allow-listed ops after the Constant->initializer rewrite. The union over 31 files is Add, Cast, Clip, Concat, Conv, Flatten, Gather, Gemm, MatMul, Mul, ReduceMax, ReduceMean, ReduceSum, Relu, Reshape, Sigmoid, Slice, Sub, Tanh, Transpose, Unsqueeze, and every shape/index input of Reshape/Gather/Slice/Unsqueeze is an initializer. These files were not run through the full `validate_submission.py`, so the I/O dim contract and the dry-run were not checked. — local benchmark `bench_arch.py`/`bench2.py`/`bench3.py` + op audit

### Inferences
- Constant nodes come from Python scalars, slice bounds and reshape shapes. `register_buffer` tensors become initializers by default (`export_params=True`), so fixed kernels and index tables are safe.
- Slicing like `g[:, :c]` exports as Slice with constant starts/ends. Keep slice bounds Python ints, never tensors derived from x.
- onnxsim is unnecessary: ORT folds and fuses at load, and BN is already folded by the exporter. onnxsim can also re-emit `Constant` nodes or change I/O dim params, so if used, run `fix_constants` and the validator afterwards (not tested here).
- Watch node and initializer budgets for feature-heavy graphs: MAX_NODES 1024 and MAX_INITIALIZERS 512. hand_v1 alone has 383 nodes. A hybrid (hand_v1 extractor + res4x48 + SE) should stay under about 600.
- Check the parity of ORT outputs vs PyTorch (max abs diff < 1e-4) on real positions after export, especially when fp16 storage or index-gather heads are used.

### Gaps
- The PyTorch TorchScript-exporter doc page returned 404 in this session. The export claims rest on submission.md, the team's c1.py and the measured exports.
