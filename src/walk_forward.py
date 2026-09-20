"""Walk-forward (rolling out-of-sample) validation.

The point: a Sharpe ratio computed on the same window used to pick a
strategy's parameters is close to meaningless (overfitting). Walk-forward
splits history into successive train/test blocks, and only ever reports
performance on data the "fit" step didn't touch.

`walk_forward()` re-runs a fixed-parameter strategy across rolling test
windows so you can see whether performance is stable through time or
concentrated in one lucky stretch. `walk_forward_with_selection()` adds
nested (train-only) parameter search on top of the same rolling-window
scaffolding -- see its docstring for why the nesting matters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


def walk_forward_with_selection(
    prices: pd.DataFrame,
    weight_fn,
    param_grid: list[dict],
    train_days: int = 252,
    test_days: int = 63,
    cost_bps: float = 10.0,
    long_only: bool = False,
) -> WalkForwardResult:
    """Nested walk-forward: at each window, pick the best parameter set from
    `param_grid` using ONLY the training slice, then apply that parameter
    set to generate the window's signal and score ONLY the held-out test
    slice.

    This is the standard fix for the failure mode where a single "best"
    parameter set is chosen by looking at performance over the whole
    history (including the test data it's then reported on) -- that is
    fitting to the test set with extra steps. Here, parameter selection
    literally cannot see test-period returns: it is computed once per
    window from `window_prices.iloc[:train_days]` alone, before the test
    slice's weights are even generated.

    Keep `param_grid` small. Every additional parameter combination tried is
    another shot at finding one that looks good on the training slice by
    chance (the multiple-comparisons problem) -- a 3x2 grid is a reasonable
    ceiling for a single-signal search; do not grid-search dozens of
    combinations and call the result robust.
    """
    n = len(prices)
    step = test_days
    start = 0
    window_records = []
    oos_returns = []

    while start + train_days + test_days <= n:
        window_end = start + train_days + test_days
        window_prices = prices.iloc[start:window_end]
        train_slice = window_prices.iloc[:train_days]

        best_params, best_train_sharpe = None, -np.inf
        for params in param_grid:
            train_weights = weight_fn(train_slice, **params)
            train_result = run_backtest(train_slice, train_weights, cost_bps=cost_bps, long_only=long_only)
            train_sharpe = train_result["stats"]["sharpe"]
            if train_sharpe > best_train_sharpe:
                best_train_sharpe, best_params = train_sharpe, params

        weights = weight_fn(window_prices, **best_params)
        result = run_backtest(window_prices, weights, cost_bps=cost_bps, long_only=long_only)

        test_start_date = window_prices.index[train_days]
        test_returns = result["returns"][result["returns"].index >= test_start_date]

        window_records.append({
            "train_start": window_prices.index[0],
            "test_start": test_start_date,
            "test_end": window_prices.index[-1],
            "selected_params": best_params,
            "train_sharpe": best_train_sharpe,
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
