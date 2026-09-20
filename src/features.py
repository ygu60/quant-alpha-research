"""
Feature engineering for the ML signal-mining study (RESEARCH_MEMO.md §12).

Deliberately "non-semantic": every feature is a generic price/volume-derived
statistic (momentum at several horizons, realized vol, an oscillator, a
moving-average gap, a volume surge ratio, rolling skew/autocorrelation, plus
same-day cross-sectional ranks of several of those). None of them are picked
because of an economic story -- the point of this study is to let a model
search for structure a hand-picked, literature-motivated signal (the rest of
this project) wouldn't be looking for. That is also exactly why this code
leans harder on purged out-of-sample validation than the rest of the
project: a feature with no theoretical reason to work has no prior evidence
against it being a pure artifact of this one sample, so the validation has
to do all the work theory would otherwise share.

Every feature is computed from a rolling window ending at day t (or is a
same-day cross-sectional statistic) -- causal by construction, same
requirement as everything else in this codebase. The FORWARD return label
(`forward_return`) is the one deliberately non-causal function here; it is a
training target, never a feature, and callers must never feed it back in as
an input (see src/ml.py's purge logic for how the boundary between "label"
and "leaked future feature" is enforced across a walk-forward split).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_autocorr_lag1(daily_returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling lag-1 autocorrelation via the covariance/variance identity
    (pandas has no built-in rolling autocorr for a DataFrame). Causal: uses
    only the trailing `window` days of returns ending at t.
    """
    lagged = daily_returns.shift(1)
    mean_r = daily_returns.rolling(window).mean()
    mean_l = lagged.rolling(window).mean()
    cov = (daily_returns * lagged).rolling(window).mean() - mean_r * mean_l
    var = daily_returns.rolling(window).var(ddof=0)
    return (cov / var.replace(0.0, np.nan)).clip(-1.0, 1.0)


def _rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = prices.diff()
    gain = delta.clip(lower=0.0).rolling(window).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_features(prices: pd.DataFrame, volume: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Compute the full non-semantic feature set. Returns a dict of
    feature-name -> DataFrame shaped exactly like `prices` (date x ticker),
    matching the "weights" convention used throughout this codebase so the
    same rolling/shift/rank idioms apply.
    """
    daily_ret = prices.pct_change()
    feats: dict[str, pd.DataFrame] = {}

    for h in (1, 5, 10, 20):
        feats[f"ret_{h}"] = prices.pct_change(h)

    for w in (10, 20):
        feats[f"vol_{w}"] = daily_ret.rolling(w).std(ddof=1)

    feats["skew_20"] = daily_ret.rolling(20).skew()
    feats["autocorr_20"] = _rolling_autocorr_lag1(daily_ret, 20)
    feats["rsi_14"] = _rsi(prices, 14)

    ma10 = prices.rolling(10).mean()
    ma50 = prices.rolling(50).mean()
    feats["ma_gap_10_50"] = (ma10 - ma50) / ma50

    bb_mean = prices.rolling(20).mean()
    bb_std = prices.rolling(20).std(ddof=1)
    feats["bb_z_20"] = (prices - bb_mean) / bb_std.replace(0.0, np.nan)

    if volume is not None:
        dollar_vol = prices * volume
        short_vol = dollar_vol.rolling(5).mean()
        long_vol = dollar_vol.rolling(20).mean()
        feats["vol_surge_5_20"] = (short_vol / long_vol.replace(0.0, np.nan)) - 1.0

    # Cross-sectional (same-day) percentile ranks of a few base features --
    # "how does this name compare to its peers today", the one place a
    # feature depends on more than just its own history. Still causal: only
    # uses information available at the close of day t.
    for base in ("ret_5", "ret_20", "vol_20"):
        feats[f"xrank_{base}"] = feats[base].rank(axis=1, pct=True)

    return feats


def forward_return(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """The supervised-learning TARGET: return from close(t) to close(t+horizon).
    Deliberately non-causal (it looks forward) -- this is a label, and must
    only ever be used as `y` in a fit() call, never concatenated into a
    feature matrix. See src/ml.py for how training samples near a
    train/test boundary are purged using this.
    """
    return prices.shift(-horizon) / prices - 1.0


def stack_to_long(frames: dict[str, pd.DataFrame], index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    """Flatten a dict of wide (date x ticker) DataFrames into one long
    DataFrame indexed by (date, ticker), one column per entry in `frames`.
    Implemented with a manual reindex + ravel (rather than DataFrame.stack)
    so behavior doesn't depend on the installed pandas version's stack/NaN
    defaults, which changed across pandas releases.
    """
    midx = pd.MultiIndex.from_product([index, columns], names=["date", "ticker"])
    data = {name: df.reindex(index=index, columns=columns).to_numpy().ravel() for name, df in frames.items()}
    return pd.DataFrame(data, index=midx)
