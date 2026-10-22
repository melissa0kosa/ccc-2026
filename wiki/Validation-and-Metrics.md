# Validation and Metrics

## Rule 1: measure with the contest metric
Compute exactly what the Acceptance Criteria slide states, on held-out data. `ccc_2026.cv.METRICS` provides `rmse`, `mae`, `accuracy` and `macro_f1`.

| Metric | Notes |
|---|---|
| RMSE | Sensitive to outliers; MSE loss optimizes it directly |
| MAE | Use L1 or Huber loss, or predict the median |
| Accuracy | Fine for balanced classes |
| **Macro-F1** | Every class counts equally, so minority classes matter. Use class weights or oversampling, and tune per-class thresholds |
| Top-k / ranking accuracy | Rank by predicted score; validate the ranking itself |

## Choosing the split
| Situation | Splitter |
|---|---|
| i.i.d. rows, regression | `KFold(5, shuffle=True)` |
| Classification | `StratifiedKFold(5, shuffle=True)` |
| Rows grouped by entity | `GroupKFold`, or aggregate to one row per entity |
| Time series / forecasting | Walk-forward: train on the past, validate on the **most recent** window. Never shuffle |
| Big data (> 1 M rows) | A single holdout of 10–20% is fine; use CV only for small data |

`ccc_2026.cv.cross_validate(model, X, y, metric, task, groups=..., time_series=...)` returns `(score, oof_predictions)`.

## Margins
- Hidden test files can be harder or shifted. **Target CV 10–20% better than the threshold.**
- If the inputs are files a–e, check the threshold on each one; some files are harder.
- If train and test distributions differ (check `describe()`), use **adversarial validation**: train a classifier to tell train from test. If its AUC is ≫ 0.5, validate on the train rows that look most like test.

## Neural network specifics
- Fit scalers on the training fold only.
- Early stopping on the validation fold already uses that fold, so validation scores are slightly optimistic. For the final model, retrain on all data for about the best number of epochs.
