# Classic ML Playbook

Rule of thumb for tabular data: **gradient boosting > random forest > NN** in accuracy per minute of effort. Start here even in a "neural network" contest unless the level says otherwise.

## Regression
| Model | When | Snippet |
|---|---|---|
| Linear / Ridge | Relationship is linear or the data is tiny. Won 2025 L2 | `make_pipeline(StandardScaler(), Ridge())` |
| Polynomial + Ridge | Smooth non-linearity, few features | `make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge())` |
| LightGBM | Default for tabular data | see below |
| CatBoost | Many categoricals; good with no tuning | `CatBoostRegressor(verbose=0)` |

If an "exact" formula might exist (physics, units), print `LinearRegression().coef_`. Clean coefficients reveal it.

## Classification
- `LogisticRegression` is the baseline.
- `RandomForestClassifier(n_estimators=300-500, n_jobs=-1)` is robust with no tuning. It won 2025 L4 (macro-F1 > 0.9).
- For imbalanced classes: `class_weight="balanced"`, `RandomOverSampler`, or per-class threshold tuning for macro-F1.

## Random forests
- Parameters that matter: `n_estimators` (more is better, until it gets slow), `max_features` (`"sqrt"` for classification, `1.0` or `0.5` for regression), `min_samples_leaf` (1–5).
- `oob_score=True` gives free validation.
- `feature_importances_` helps you prune.

## Gradient boosting (LightGBM / XGBoost / CatBoost)
```python
import lightgbm as lgb
m = lgb.LGBMRegressor(n_estimators=5000, learning_rate=0.03, num_leaves=63,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      min_child_samples=20, reg_lambda=1.0, verbose=-1)
m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(200)])
```
- Order of tuning:
  1. `learning_rate` low with early stopping
  2. `num_leaves` / `max_depth`
  3. `min_child_samples`
  4. subsampling
  5. regularization
- XGBoost on GPU: `XGBRegressor(device="cuda", tree_method="hist")`. CatBoost on GPU: `task_type="GPU"`.
- Categoricals: LightGBM accepts the `category` dtype; CatBoost takes `cat_features=[...]`.

## Hyperparameter search (Optuna)
```python
import optuna
def objective(t):
    p = dict(num_leaves=t.suggest_int("num_leaves", 15, 255, log=True),
             learning_rate=t.suggest_float("lr", 0.01, 0.2, log=True),
             min_child_samples=t.suggest_int("mcs", 5, 100),
             colsample_bytree=t.suggest_float("cs", 0.5, 1.0))
    return cross_validate(lgb.LGBMRegressor(n_estimators=500, verbose=-1, **p),
                          X, y, "rmse", "regression", verbose=False)[0]
study = optuna.create_study(direction="minimize"); study.optimize(objective, timeout=300)
```
Always use `timeout=` in a contest.

## Ensembling
- Average the predictions of LightGBM, CatBoost and an NN. Diverse models give the biggest gain.
- Stacking: fit a Ridge on the out-of-fold predictions. The 2023 top-10% solution was DNN + XGBoost + LightGBM stacked.
- Seed averaging: train 3–5 seeds of the same model and average.

## Time series / forecasting
- Lag features (`y.shift(k)`), rolling means and calendar features, with a walk-forward split.
- For "top-k next days" tasks, framing it as ranking or classification often beats raw regression (2025 L5).
