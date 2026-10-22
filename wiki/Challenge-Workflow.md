# Challenge Workflow (per level)

Time is the tiebreaker, so follow this order and skip nothing.

## 0. Set up (1 min)
```bash
uv run ccc new 3
uv run ccc unzip path/to/level_3.zip 3
```
Put the PDF (or the transcript of a video/audio statement) into `levels/level3/`.

## 1. Read the statement (5 min)
Write `levels/levelN/STATEMENT.md` with:
- **Task:** what to predict or compute.
- **Input format:** columns, units, header (yes/no), separators.
- **Output format:** exact columns, order, header, decimals, row order.
- **Metric and threshold:** the "Acceptance Criteria" slide. Note it per file (2025 L2: RMSE < 6 on file a, < 3 on b and c).
- **Gotchas the slides hint at:** units, spelled-out numbers, missing values.

## 2. Reproduce the sample first
Make `solve.py` produce `*_sample.out` exactly. The template prints `SAMPLE OK` or a diff. Format bugs cost penalty minutes, and modelling can't fix them.

## 3. Look at the data (5–10 min)
Work through the [Data Cleaning Checklist](Data-Cleaning-Checklist.md). Before modelling, ask: **is there a shortcut?**
- Exact / analysis levels (sort, count, PCA std, unit conversion) need no model.
- Leaks: ids that encode the label, rows duplicated between train and test, test equal to a mirrored or shifted copy of train (2025 L6 was a time-reversed copy of L5).
- Hidden puzzles: predictions reshaped into an image spell a password (2023 L2 and L7).

## 4. Baseline fast (≤ 10 min)
```python
from ccc_2026.cv import run_baselines
run_baselines(X, y, metric="rmse", task="regression")   # ridge, RF, HistGB, LightGBM
```
If a baseline already beats the threshold **with margin**, submit and move on.

## 5. Improve only as much as needed
- Tabular: feature engineering, then LightGBM/CatBoost tuning, then an ensemble. See [Classic ML Playbook](Classic-ML-Playbook.md).
- Sequences, images, audio, or "build a neural network" levels: see [Neural Networks Playbook](Neural-Networks-Playbook.md).
- Aim for CV **10–20% better than the threshold**, because hidden graders can shift.

## 6. Predict, check, upload
- Train the final model on **all** labelled data, using the hyperparameters chosen by CV.
- Re-run the sample check. Spot-check one output file by eye: row count, header, ids, NaNs.
- Upload all outputs, then **upload the source code** (likely a time bonus).
- `git commit -am "level N"` so teammates can pull.

## Team split (2–3 people)
- **Person A:** reads the statement, writes the parser and output writer, checks the sample.
- **Person B:** does EDA and cleaning, finds shortcuts and leaks.
- **Person C:** runs baselines and NN experiments as soon as `X, y` exist.
- Read the next level's PDF as soon as it unlocks while the others finish the current one.
