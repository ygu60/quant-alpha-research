"""
Purged walk-forward machine-learning signal search (RESEARCH_MEMO.md §12).

This is a different, stricter validation loop than src/walk_forward.py, not
a reuse of it, because ML introduces a leakage path the simpler strategies
don't have: the training TARGET is a forward-looking return
(`features.forward_return`), so a training sample near the end of a
training window has a label that reaches past the train/test boundary into
the test period it's supposed to be evaluated on. Walking that boundary
back by the label horizon ("purging") is standard practice in financial ML
(Lopez de Prado, *Advances in Financial Machine Learning*, 2018, ch. 7) and
is the one thing this module exists to get right; everything else
(features causal, weights shifted forward one day before backtest.py
applies them) reuses the same conventions as the rest of this project.

Per walk-forward window:
  1. Compute features + forward-return labels ONCE globally (both are pure
     functions of trailing/leading windows around each day -- computing
     them on the full history and then slicing per window gives identical
     values to recomputing per window, just without redoing the rolling-
     window arithmetic 37 times).
  2. Training rows = the window's first `train_days`, MINUS the last
     `horizon` days (purge) -- those rows' labels reach into the test
     period.
  3. Fit a fresh, unfit model (from `build_model()`) on the purged training
     rows. A fresh model every window, same as `weight_fn` being called
     fresh per window elsewhere in this project -- no state carries across
     windows.
  4. Predict on the test period's features (test labels are NEVER passed to
     the model -- they're only used afterward, by run_backtest, exactly the
     same way every other strategy's future returns are only used for
     scoring, not for generating the signal).
  5. Rank predictions cross-sectionally each test day, go long the top
     `top_frac` (equal-weighted, long-only), score via the same
     `run_backtest` used everywhere else in this project.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .features import compute_features, forward_return, stack_to_long
from .metrics import summarize


@dataclass
class MLWalkForwardResult:
    window_stats: pd.DataFrame
    combined_returns: pd.Series
    combined_stats: dict
    predictions: pd.DataFrame  # columns: date, ticker, predicted, actual -- for calibration plots
    feature_importances: pd.DataFrame  # one row per window, one column per feature (NaN if model has none)


def run_ml_walk_forward(
    prices: pd.DataFrame,
    volume: pd.DataFrame | None,
    build_model,
    horizon: int = 5,
    train_days: int = 252,
    test_days: int = 63,
    top_frac: float = 0.20,
    cost_bps: float = 10.0,
    min_train_rows: int = 200,
    shuffle_labels: bool = False,
    random_state: int = 0,
    risk_overlay=None,
) -> MLWalkForwardResult:
    """`build_model()` must return a FRESH, unfit scikit-learn-style
    regressor (called once per window). `shuffle_labels=True` runs the
    exact same pipeline with training labels randomly permuted within each
    window -- a negative control: if this ever produces a real out-of-
    sample edge, that's evidence of a leakage bug somewhere in this module,
    not of a strategy (see tests/test_ml.py). `risk_overlay`, if given, is
    called as `risk_overlay(window_prices, full_weights) -> weights` right
    before scoring (e.g. `src.risk.apply_full_overlay` partially applied) --
    same non-negative, already-long-only weights convention as the rest of
    this project, so the result must be scored with the same long_only=False
    used here.
    """
    feature_frames = compute_features(prices, volume)
    labels = forward_return(prices, horizon)
    feature_cols = list(feature_frames.keys())

    long_df = stack_to_long({**feature_frames, "label": labels}, prices.index, prices.columns)
    date_level = long_df.index.get_level_values("date")
    rng = np.random.default_rng(random_state)

    n = len(prices)
    step = test_days
    start = 0
    window_records = []
    oos_returns = []
    importances = []
    pred_records = []

    while start + train_days + test_days <= n:
        window_dates = prices.index[start:start + train_days + test_days]
        train_dates = window_dates[:train_days]
        test_dates = window_dates[train_days:]
        purged_train_dates = train_dates[:-horizon] if horizon > 0 else train_dates

        train_mask = date_level.isin(purged_train_dates)
        test_mask = date_level.isin(test_dates)

        train_data = long_df.loc[train_mask].dropna(subset=feature_cols + ["label"])
        test_data = long_df.loc[test_mask].dropna(subset=feature_cols)

        if len(train_data) < min_train_rows or len(test_data) == 0:
            start += step
            continue

        y_train = train_data["label"].to_numpy()
        if shuffle_labels:
            y_train = rng.permutation(y_train)

        model = build_model()
        model.fit(train_data[feature_cols].to_numpy(), y_train)
        preds = model.predict(test_data[feature_cols].to_numpy())

        pred_series = pd.Series(preds, index=test_data.index)
        pred_wide = pred_series.unstack("ticker").reindex(index=test_dates, columns=prices.columns)

        n_assets_available = pred_wide.notna().sum(axis=1)
        k = np.maximum(1, np.round(n_assets_available * top_frac)).astype(int)
        ranks = pred_wide.rank(axis=1, ascending=False, method="first")
        window_weights = pd.DataFrame(0.0, index=test_dates, columns=prices.columns)
        for dt in test_dates:
            kk = k.loc[dt]
            if kk == 0:
                continue
            mask = ranks.loc[dt] <= kk
            window_weights.loc[dt, mask] = 1.0 / kk

        full_weights = pd.DataFrame(0.0, index=window_dates, columns=prices.columns)
        full_weights.loc[test_dates] = window_weights
        window_prices = prices.loc[window_dates]
        if risk_overlay is not None:
            full_weights = risk_overlay(window_prices, full_weights)
        result = run_backtest(window_prices, full_weights, cost_bps=cost_bps, long_only=False)
        test_returns = result["returns"][result["returns"].index >= test_dates[0]]

        window_records.append({
            "train_start": window_dates[0],
            "test_start": test_dates[0],
            "test_end": window_dates[-1],
            "n_train_rows": len(train_data),
            "oos_sharpe": summarize(test_returns)["sharpe"] if len(test_returns) > 1 else float("nan"),
            "oos_return": float((1 + test_returns).prod() - 1),
        })
        oos_returns.append(test_returns)

        if hasattr(model, "feature_importances_"):
            importances.append(dict(zip(feature_cols, model.feature_importances_)))
        elif hasattr(model, "coef_"):
            importances.append(dict(zip(feature_cols, np.abs(np.ravel(model.coef_)))))

        actual = test_data["label"]
        common_idx = pred_series.index.intersection(actual.index)
        pred_records.append(pd.DataFrame({
            "predicted": pred_series.loc[common_idx],
            "actual": actual.loc[common_idx],
        }))

        start += step

    combined = pd.concat(oos_returns).sort_index() if oos_returns else pd.Series(dtype=float)
    combined_stats = summarize(combined) if len(combined) else {}
    predictions = pd.concat(pred_records) if pred_records else pd.DataFrame(columns=["predicted", "actual"])
    importances_df = pd.DataFrame(importances) if importances else pd.DataFrame(columns=feature_cols)

    return MLWalkForwardResult(
        window_stats=pd.DataFrame(window_records),
        combined_returns=combined,
        combined_stats=combined_stats,
        predictions=predictions,
        feature_importances=importances_df,
    )
