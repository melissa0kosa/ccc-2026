"""Cross-validation harness and quick baselines for tabular levels."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

METRICS: dict[str, Callable] = {
    "rmse": root_mean_squared_error,
    "mae": mean_absolute_error,
    "accuracy": accuracy_score,
    "macro_f1": lambda y, p: f1_score(y, p, average="macro"),
}
LOWER_IS_BETTER = {"rmse", "mae"}


def make_splitter(task: str, n_splits: int = 5, groups=None, time_series: bool = False,
                  seed: int = 0):
    if time_series:
        return TimeSeriesSplit(n_splits=n_splits)
    if groups is not None:
        return GroupKFold(n_splits=n_splits)
    if task == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def cross_validate(model, X, y, metric: str, task: str, n_splits: int = 5, groups=None,
                   time_series: bool = False, seed: int = 0, verbose: bool = True):
    """Out-of-fold predictions + per-fold scores using exactly the contest metric."""
    X, y = np.asarray(X), np.asarray(y)
    score_fn = METRICS[metric]
    oof = np.empty(len(y), dtype=object if task == "classification" else float)
    fitted = np.zeros(len(y), dtype=bool)
    scores = []
    splitter = make_splitter(task, n_splits, groups, time_series, seed)
    for fold, (tr, va) in enumerate(splitter.split(X, y, groups)):
        m = clone(model).fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
        fitted[va] = True
        scores.append(score_fn(y[va], oof[va]))
        if verbose:
            print(f"  fold {fold}: {metric}={scores[-1]:.4f}")
    total = score_fn(y[fitted], oof[fitted])
    if verbose:
        print(f"  {type(model).__name__}: {metric}={total:.4f} (+/- {np.std(scores):.4f})")
    return total, oof


def baselines(task: str, seed: int = 0) -> dict:
    """A small zoo of strong, fast defaults. Try all, keep the best, then tune that one."""
    import lightgbm as lgb
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if task == "regression":
        return {
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "random_forest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed),
            "hist_gb": HistGradientBoostingRegressor(random_state=seed),
            "lightgbm": lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=63,
                                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                          random_state=seed, verbose=-1),
        }
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
        "hist_gb": HistGradientBoostingClassifier(random_state=seed),
        "lightgbm": lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=63,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                       random_state=seed, verbose=-1),
    }


def run_baselines(X, y, metric: str, task: str, **cv_kwargs) -> dict[str, float]:
    results = {}
    for name, model in baselines(task).items():
        print(f"[{name}]")
        results[name], _ = cross_validate(model, X, y, metric, task, **cv_kwargs)
    best = (min if metric in LOWER_IS_BETTER else max)(results, key=results.get)
    print(f"best: {best} ({metric}={results[best]:.4f})")
    return results
