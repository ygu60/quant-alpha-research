"""
Market-regime detection, requested to test whether existing signals are
regime-dependent (e.g. mean-reversion working best in high-volatility/
range-bound conditions, trend-following working best when a trend is
actually present) rather than uniformly present or absent across all of
history.

Two independent approaches, both causal by construction:

1. Rule-based regimes (`trend_regime`, `volatility_regime`,
   `combined_rule_regime`) -- simple rolling statistics, exactly as causal
   as every other feature in this project (a trailing window ending at t).
   No model-fitting, so no walk-forward subtlety at all.

2. Markov-switching regime (`markov_regime_walk_forward`) -- a proper
   latent-state model (Hamilton, 1989, "A New Approach to the Economic
   Analysis of Nonstationary Time Series and the Business Cycle") fit via
   `statsmodels.tsa.regime_switching.markov_regression.MarkovRegression`.
   This is NOT as simple to keep walk-forward-safe as a rolling statistic:
   the model's parameters (regime means, variances, transition
   probabilities) are estimated by maximum likelihood over WHATEVER sample
   is handed to `.fit()`, so fitting on a window that includes the test
   period would leak test-period information into the very definition of
   "which regime were we in" during that period.

   The fix mirrors this project's pair-selection pattern (src/pairs.py):
   estimate parameters ONCE per walk-forward window using ONLY the training
   slice (`fit_markov_params`), then run the model's `.filter(params)` --
   NOT `.fit()` -- across the full window with those parameters held fixed.
   `.filter()` computes the Hamilton filter (causal: the probability at day
   t uses only data through day t) without re-estimating anything, which
   was verified empirically before writing this module: refitting with
   `maxiter=0` was tried first and rejected because it still measurably
   moved the parameters (a few percent) using data past the train cutoff --
   `.filter()` with fixed params moves them by construction not at all.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def trend_regime(prices: pd.Series, window: int = 200) -> pd.Series:
    """True = price above its trailing `window`-day moving average ("trend
    up"), False otherwise. The simplest possible causal trend filter.
    """
    ma = prices.rolling(window).mean()
    return prices > ma


def volatility_regime(
    returns: pd.Series,
    vol_window: int = 20,
    percentile_window: int = 504,
    low: float = 1 / 3,
    high: float = 2 / 3,
) -> pd.Series:
    """Bucket each day's trailing `vol_window`-day realized vol into
    "low" / "mid" / "high" based on where it ranks against its own trailing
    `percentile_window`-day history (~2 years by default) -- a
    self-relative threshold rather than a fixed absolute vol level, so it
    adapts across assets/eras without being fit to this specific sample.
    """
    realized_vol = returns.rolling(vol_window).std(ddof=1)

    def _percentile_of_last(window_vals: np.ndarray) -> float:
        last = window_vals[-1]
        return float((window_vals < last).mean())

    pct_rank = realized_vol.rolling(percentile_window).apply(_percentile_of_last, raw=True)
    bucket = pd.cut(pct_rank, bins=[-0.01, low, high, 1.01], labels=["low", "mid", "high"])
    return bucket


def combined_rule_regime(prices: pd.Series, trend_window: int = 200, vol_window: int = 20,
                          percentile_window: int = 504) -> pd.Series:
    """Cross of trend (up/down) x volatility (low/mid/high) into a single
    categorical label, e.g. "up_high" = uptrend during a high-vol spell.
    """
    returns = prices.pct_change(fill_method=None)
    trend = trend_regime(prices, trend_window).map({True: "up", False: "down"})
    vol = volatility_regime(returns, vol_window, percentile_window)
    combined = trend.astype(str) + "_" + vol.astype(str)
    combined[trend.isna() | vol.isna()] = np.nan
    return combined


def fit_markov_params(train_returns: pd.Series, k_regimes: int = 2) -> pd.Series | None:
    """Fit a k-regime Markov-switching mean+variance model on TRAINING
    returns only. Returns None (rather than raising) if the MLE fails to
    converge on this particular training window -- a 252-observation
    sample is a small one for a 6-parameter (k=2) model, and a handful of
    non-convergent windows should be skipped by the caller, not crash a
    37-window study.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = MarkovRegression(train_returns * 100, k_regimes=k_regimes, trend="c", switching_variance=True)
            res = model.fit(maxiter=200)
        except Exception:
            return None
    if not np.all(np.isfinite(res.params.to_numpy())):
        return None
    return res.params


def high_vol_state_index(params: pd.Series, k_regimes: int = 2) -> int:
    """Which fitted state has the larger variance -- the "high volatility"
    regime, regardless of which raw index the optimizer happened to assign
    it (state order isn't guaranteed consistent across independent fits).
    """
    sigma2 = [params[f"sigma2[{i}]"] for i in range(k_regimes)]
    return int(np.argmax(sigma2))


def filter_high_vol_probability(window_returns: pd.Series, params: pd.Series, k_regimes: int = 2) -> pd.Series:
    """Run the Hamilton filter across `window_returns` with parameters held
    FIXED at `params` (estimated from training data elsewhere -- this
    function does no estimation at all), returning the filtered
    (causal-by-construction) probability of being in the high-volatility
    state on each day.
    """
    hv_state = high_vol_state_index(params, k_regimes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MarkovRegression(window_returns * 100, k_regimes=k_regimes, trend="c", switching_variance=True)
        filtered = model.filter(params.to_numpy())
    return filtered.filtered_marginal_probabilities[hv_state].clip(0.0, 1.0)


def markov_regime_walk_forward(
    returns: pd.Series,
    train_days: int = 252,
    test_days: int = 63,
    k_regimes: int = 2,
) -> pd.Series:
    """Walk-forward, leakage-safe high-volatility-state probability for the
    full history: for each non-overlapping test window, fit params on the
    preceding train slice, filter (fixed params) across train+test, and
    keep only the test-period probabilities -- exactly the same
    windowing convention as src/walk_forward.py, so this series lines up
    with every other walk-forward result in this project and can be
    reindexed directly against strategy returns.

    Windows where the training fit doesn't converge contribute NaN for
    that test period rather than a guess.
    """
    n = len(returns)
    step = test_days
    start = 0
    chunks = []

    while start + train_days + test_days <= n:
        train = returns.iloc[start:start + train_days]
        window = returns.iloc[start:start + train_days + test_days]
        test_index = window.index[train_days:]

        params = fit_markov_params(train, k_regimes)
        if params is None:
            chunks.append(pd.Series(np.nan, index=test_index))
        else:
            filtered = filter_high_vol_probability(window, params, k_regimes)
            chunks.append(filtered.loc[test_index])
        start += step

    return pd.concat(chunks).sort_index() if chunks else pd.Series(dtype=float)
