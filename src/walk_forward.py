"""Walk-forward (rolling out-of-sample) validation.

The point: a Sharpe ratio computed on the same window used to pick a
strategy's parameters is close to meaningless (overfitting). Walk-forward
splits history into successive train/test blocks, and only ever reports
performance on data the "fit" step didn't touch.

This module is deliberately simple: it doesn't do parameter search (no
scipy/sklearn available in this sandbox), it just re-runs a fixed-parameter
strategy across rolling test windows so you can see whether performance is
stable through time or concentrated in one lucky stretch.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import run_backtest


@dataclass
class WalkForwardResult:
    window_stats: pd.DataFrame
    combined_returns: pd.Series
    combined_stats: dict


def walk_forward(
    prices: pd.DataFrame,
    weight_fn,
    train_days: int = 252,
    test_days: int = 63,
    cost_bps: float = 10.0,
    long_only: bool = False,
    **weight_fn_kwargs,
) -> WalkForwardResult:
    """Roll a (train_days + test_days) window across `prices`.

    `weight_fn(prices_slice, **kwargs) -> weights_slice` is recomputed on
    each expanding slice up through the end of the test window (so signals
    at time t still only use data up to t), but ONLY the test_days portion
    of each window's returns is kept for the report.
    """
    n = len(prices)
    step = test_days
    start = 0
    window_records = []
    oos_returns = []

    while start + train_days + test_days <= n:
        window_end = start + train_days + test_days
        window_prices = prices.iloc[start:window_end]

        weights = weight_fn(window_prices, **weight_fn_kwargs)
        result = run_backtest(window_prices, weights, cost_bps=cost_bps, long_only=long_only)

        test_start_date = window_prices.index[train_days]
        test_returns = result["returns"][result["returns"].index >= test_start_date]

        window_records.append({
            "train_start": window_prices.index[0],
            "test_start": test_start_date,
            "test_end": window_prices.index[-1],
            "oos_sharpe": _sharpe(test_returns),
            "oos_return": float((1 + test_returns).prod() - 1),
        })
        oos_returns.append(test_returns)
        start += step

    combined = pd.concat(oos_returns).sort_index() if oos_returns else pd.Series(dtype=float)
    from . import metrics as m
    combined_stats = m.summarize(combined) if len(combined) else {}

    return WalkForwardResult(
        window_stats=pd.DataFrame(window_records),
        combined_returns=combined,
        combined_stats=combined_stats,
    )


def _sharpe(returns: pd.Series) -> float:
    from . import metrics as m
    return m.sharpe_ratio(returns) if len(returns) > 1 else float("nan")
