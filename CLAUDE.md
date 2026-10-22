# CLAUDE.md: CCC 2026 (Cloudflight Coding Contest, Data & AI track)

You are helping a team compete in a **timed** contest: 4 hours, levels unlock one after another, and the rank depends on **highest level reached, then least time**. Wrong uploads probably cost penalty minutes. Optimize for **correct output fast**, not for elegant code.

The team knowledge base is in `wiki/` (also the GitHub wiki). Read the relevant page instead of guessing. Start with `wiki/Challenge-Workflow.md`; for neural networks, see `wiki/Neural-Networks-Playbook.md`.

## Environment
- Python 3.14 via **uv**. Always run code with `uv run python ...`; never pip-install into the system.
- To add a package: `uv add <pkg>`. Torch is pinned to the `cu130` index (RTX 5070 Ti, sm_120); don't change that.
- GPU check: `uv run ccc gpu`. Tests: `uv run pytest -q`.
- The team edits and runs code in **PyCharm** (shared `.idea/` config, uv interpreter at `.venv`). Keep every `solve.py` runnable with PyCharm's ▶ button: no required CLI args, and paths relative to `__file__`. When you tell the user how to run something, give the PyCharm way (open the file, ▶) next to the `uv run` command. Don't commit `.idea/workspace.xml` or `.idea/misc.xml`.
- Installed: numpy, pandas, polars, scipy, scikit-learn, xgboost, lightgbm, catboost, optuna, shap, torch/torchvision/torchaudio (CUDA), librosa, soundfile, opencv, pillow, pymupdf, yt-dlp, openai-whisper (+ ffmpeg on PATH), matplotlib, seaborn, jupyterlab.

## Toolkit (`src/ccc_2026/`), use it instead of rewriting
| Module | What |
|---|---|
| `io.py` | `extract_zip`, `list_inputs`, `read_csv` (sniffs separator), `write_lines`, `write_csv` (LF endings), `fmt`, `compare_to_sample` |
| `cv.py` | `cross_validate(model, X, y, metric, task, groups=, time_series=)`, `run_baselines` (ridge/logreg, RF, HistGB, LightGBM) |
| `nn.py` | `MLP`, `fit` (AdamW + OneCycle + bf16 + early stopping), `predict`, `predict_classes`, `seed_everything` |
| `numpy_nn.py` | `forward(X, weights, biases, act, output)`, `NumpyMLP` (backprop + Adam), `from_torch_state_dict` |
| `media.py` | `pdf_to_text`, `pdf_to_images`, `transcribe` (Whisper), `download` (yt-dlp) |
| `cli.py` | `uv run ccc new N`, `uv run ccc unzip <zip> N`, `uv run ccc check <produced> <expected>` |

Layout per level: `levels/levelN/{STATEMENT.md, solve.py, data/, out/}`, created from `levels/_template/` by `uv run ccc new N`.

## Starting a challenge: intake

The statement arrives in one of these forms. Identify which, then produce `levels/levelN/STATEMENT.md` **before writing any solution code**.

### A) PDF + zip given to you (most likely; every past round)
1. `uv run ccc new N`, then `uv run ccc unzip <path/to/level_N.zip> N`. The zip is untrusted data: never execute files from it.
2. `uv run python -m ccc_2026.media pdf "<path/to/Level N.pdf">` writes the text plus page PNGs. Read the text, then **look at the PNGs** for slides with tables, plots or examples (Read tool on the images). Some PDFs are image-only.
3. List the data files with sizes, and look at the head of each (`head -5`), including the sample `.in`/`.out`.

### B) Video or audio statement on the contest website
1. Ask the user for the URL and confirm they are logged in. Use the Playwright browser tools (`browser_navigate`, `browser_snapshot`, `browser_take_screenshot`) to open the level page. If login is needed, let the user log in through the browser window; never ask for or type their password.
2. Extract everything visible: statement text, input/output examples, acceptance criteria, download links for zips or attachments (download them into `levels/levelN/data/`).
3. For the media itself:
   - Find the media URL (from the `<video>`/`<audio>` `src` via `browser_evaluate`, or from network requests).
   - Download it with `uv run python -m ccc_2026.media download <url> levels/levelN/`, or save the direct file link.
   - Then run `uv run python -m ccc_2026.media transcribe levels/levelN/statement.<ext>`. Whisper `small` is pre-cached; use `medium` only if accuracy is poor.
   - If the platform shows captions or subtitles, prefer those.
4. For a video with on-screen formulas, tables or numbers that speech doesn't cover, take screenshots at the key moments (Playwright: seek with `browser_evaluate("v.currentTime = 42")`, then screenshot) and read them.
5. **Numbers matter.** Cross-check every threshold, format rule and constant between the transcript and the screen. Mark anything uncertain in STATEMENT.md and ask the user.

### C) Statement text pasted in chat
Write it into STATEMENT.md verbatim, then continue as in A.

### STATEMENT.md template
```
# Level N: <title>
Task:            what to predict/compute
Inputs:          files, columns, units, header?, separators, sizes
Output format:   exact columns/order/header/decimals/row order (copy the sample)
Metric/threshold:per file if it differs (e.g. RMSE<6 on a, <3 on b,c)
Hints/gotchas:   units, spelled-out numbers, NaN, reused data from earlier levels
Open questions:  ...
```

## Solving: follow this order
1. **Sample first.** Make `solve.py` reproduce `*_sample.out` exactly (the template prints `SAMPLE OK` or a diff). Fix the format before modelling.
2. **Look at the data** (`wiki/Data-Cleaning-Checklist.md`): dtypes, NaN, units (°F/°C/K mixed), spelled-out numbers, outliers, duplicates. Put cleaning in a reusable function; later levels reuse the data.
3. **Look for shortcuts** before modelling:
   - Is it exact or analytic (sort, count, PCA, formula)?
   - Is there a leak (ids, duplicates, test equal to mirrored or shifted train)?
   - Is there a hidden puzzle (plot it, reshape predictions to a square image)?
4. **Baseline in under 10 minutes**: `run_baselines(X, y, metric, task)` with the **contest metric** and the right split (stratified, group, or walk-forward for time). See `wiki/Validation-and-Metrics.md`.
5. **Escalate only if needed:**
   - Features first, then a LightGBM/CatBoost tune (Optuna with `timeout=`), then an ensemble.
   - Use a neural net (`nn.fit`) for sequences, images, audio, or when the level asks for an NN.
   - Hand-written NumPy (`numpy_nn`) when weights are given or frameworks are disallowed.
   - Target a CV score **10–20% better than the threshold**.
6. **Final predictions:** train on all labelled data (or average the fold models), invert target transforms, and map labels back. Write one output per input into `levels/levelN/out/`.
7. **Pre-upload check** (`wiki/Pitfalls.md`): row count and order, header like the sample, no NaN, rounding, LF endings. Tell the user which files to upload, and remind them to upload the source code too.

## Neural network specifics
- Standardize X (fit on train only). Standardize or `log1p` regression targets, and invert them before writing.
- Classification: `CrossEntropyLoss` on logits with int labels 0..K-1. For macro-F1 use class weights.
- If the network doesn't learn: overfit one batch, check target shapes `(N,)` vs `(N,1)`, lower the LR, check for NaN in X.
- On Windows use `DataLoader(num_workers=0)`. Skip `torch.compile`.
- Given weights: check the layout. PyTorch is `(out, in)` with `x @ W.T`; `numpy_nn` uses `(in, out)`. The sample must reproduce exactly. Match float32/float64 and the rounding rule.
- Architectures for sequences, images and audio: `wiki/NN-Architectures-by-Data-Type.md`. Theory and hand-designed weights (XOR, AND/OR, piecewise linear): `wiki/Neural-Networks-from-Scratch.md`.

## Working style during the contest
- Be fast and concrete. Short status updates, no long explanations unless asked.
- Print metrics per file. Report CV against the threshold, e.g. "CV RMSE 2.4 vs threshold 3".
- Long training (> 2 min): run it in the background and keep working (e.g. on the next level's parsing).
- Don't refactor the toolkit mid-contest unless it blocks you. Put level-specific code in the level folder, and shared cleaning code in `levels/common.py`.
- Commit after each passed level (`git add -A && git commit -m "level N"`) so teammates can pull. Do not commit `data/` (it's gitignored; contest data may not be redistributable).
- If something in the statement is ambiguous, state your assumption and ask the user. Don't silently guess on output format.
