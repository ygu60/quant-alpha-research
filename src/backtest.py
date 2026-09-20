"""Vectorized backtest engine with explicit look-ahead protection and costs."""
from __future__ import annotations

import pandas as pd

from . import metrics as m


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float = 10.0,
    long_only: bool = False,
) -> dict:
    """Apply target weights to next-day returns and compute performance.

    Parameters
    ----------
    prices : DataFrame of close prices, one column per asset.
    weights : DataFrame, same shape/index as prices. weights.loc[t] is the
        target portfolio computed using information available AT THE CLOSE
        of day t. It is applied to the return realized from close(t) to
        close(t+1) -- i.e. shifted forward by one day here, so a strategy
        cannot "trade on" a return it hasn't seen yet.
    cost_bps : one-way transaction cost in basis points, charged on the
        change in weight each day (a simple stand-in for bid-ask spread +
        slippage; Robinhood charges no commission, but spread/slippage is
        real and matters for small-name strategies).
    long_only : if True, negative weights are clipped to zero and the book
        is renormalized -- use this to model a plain cash account that
        cannot short individual equities.
    """
    if long_only:
        weights = weights.clip(lower=0.0)
        row_sums = weights.sum(axis=1)
        safe_sums = row_sums.mask(row_sums == 0, 1.0)
        weights = weights.div(safe_sums, axis=0)
        weights = weights.mask(row_sums == 0, 0.0)

    asset_returns = prices.pct_change(fill_method=None)
    applied_weights = weights.shift(1).fillna(0.0)  # decided at t, applied to t->t+1 return

    gross_returns = (applied_weights * asset_returns).sum(axis=1)

    turnover_series = applied_weights.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover_series * (cost_bps / 10_000.0)

    net_returns = gross_returns - costs
    net_returns = net_returns.iloc[1:]  # drop the first NaN return row

    stats = m.summarize(net_returns, weights=applied_weights.loc[net_returns.index])
    equity_curve = (1 + net_returns).cumprod()

    return {
        "returns": net_returns,
        "equity_curve": equity_curve,
        "weights": applied_weights,
        "stats": stats,
    }
