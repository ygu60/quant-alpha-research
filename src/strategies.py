"""
Candidate statistical strategies.

Every strategy exposes `generate_weights(prices, **params) -> pd.DataFrame`
returning a target portfolio weight per asset per day, using ONLY information
available up to and including that day's close. `backtest.py` shifts weights
forward by one day before applying them to returns, so there is no look-ahead.

These are chosen specifically because they are "capacity-constrained" --
academically documented anomalies whose alpha decays as the dollars trying to
exploit them grow, which is exactly the regime a single-thousands-dollar
account sits in and a multi-billion-dollar fund cannot:

  - Short-term reversal (Lehmann 1990; Jegadeesh 1990): after an outsized
    1-5 day move, small/illiquid names show partial mean reversion. Large
    funds can't harvest this at scale -- the trade size needed to matter to
    their P&L moves the price before they can complete it.
  - Pairs / stat-arb on small, correlated names: spreads are thin in dollar
    terms; a fund needs many pairs at large size to matter, a small account
    needs just a few.
  - Seasonal / calendar effects (turn-of-month, day-of-week): well documented,
    small in magnitude, easily swamped by costs at size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def short_term_reversal(prices: pd.DataFrame, lookback: int = 3, n_long: int = 2, n_short: int = 2) -> pd.DataFrame:
    """Cross-sectional short-term reversal.

    Rank assets by their trailing `lookback`-day return. Go long the biggest
    losers, short the biggest winners (in a cash account, "short" just means
    "underweight / avoid" -- see backtest.py long_only flag).
    """
    rets = prices.pct_change(lookback, fill_method=None)
    ranks = rets.rank(axis=1, method="first")
    n_assets = prices.shape[1]

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    long_mask = ranks.le(n_long)          # worst performers -> expect bounce
    short_mask = ranks.gt(n_assets - n_short)  # best performers -> expect fade

    weights[long_mask] = 1.0 / n_long
    weights[short_mask] = -1.0 / n_short
    weights[rets.isna()] = 0.0
    return weights


def zscore_pairs(prices: pd.DataFrame, asset_a: str, asset_b: str, lookback: int = 20, entry_z: float = 1.5, exit_z: float = 0.3) -> pd.DataFrame:
    """Mean-reversion pairs trade on the price-ratio spread of two assets.

    Long the spread (long A, short B) when it's cheap (z < -entry_z),
    short the spread when it's rich (z > entry_z), flatten near zero.
    """
    spread = np.log(prices[asset_a]) - np.log(prices[asset_b])
    mu = spread.rolling(lookback).mean()
    sigma = spread.rolling(lookback).std(ddof=1)
    z = (spread - mu) / sigma

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    position = pd.Series(0.0, index=prices.index)  # +1 = long spread, -1 = short spread

    state = 0.0
    for t in range(len(z)):
        zt = z.iloc[t]
        if np.isnan(zt):
            position.iloc[t] = state
            continue
        if state == 0.0:
            if zt < -entry_z:
                state = 1.0
            elif zt > entry_z:
                state = -1.0
        else:
            if abs(zt) < exit_z:
                state = 0.0
        position.iloc[t] = state

    weights[asset_a] = 0.5 * position
    weights[asset_b] = -0.5 * position
    return weights


def turn_of_month(prices: pd.DataFrame, days_before: int = 1, days_after: int = 3) -> pd.DataFrame:
    """Equal-weight long book, but only "on" during the turn-of-month window
    (a well-documented small calendar anomaly: last `days_before` trading days
    of the month through the first `days_after` of the next month).
    """
    idx = prices.index
    month = idx.to_series().dt.to_period("M")
    is_month_end = month.ne(month.shift(-1))
    is_month_start = month.ne(month.shift(1))

    on = pd.Series(False, index=idx)
    end_positions = np.where(is_month_end.values)[0]
    start_positions = np.where(is_month_start.values)[0]

    for pos in end_positions:
        on.iloc[max(0, pos - days_before + 1): pos + 1] = True
    for pos in start_positions:
        on.iloc[pos: min(len(idx), pos + days_after)] = True

    n_assets = prices.shape[1]
    weights = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    weights.loc[on, :] = 1.0 / n_assets
    return weights
