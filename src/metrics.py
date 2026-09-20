"""Performance metrics for backtested return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    excess = returns - rf / periods_per_year
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    excess = returns - rf / periods_per_year
    downside = excess[excess < 0]
    dd_std = downside.std(ddof=1)
    if dd_std == 0 or np.isnan(dd_std):
        return 0.0
    return float(excess.mean() / dd_std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    growth = (1 + returns).prod()
    n = len(returns)
    if n == 0:
        return 0.0
    return float(growth ** (periods_per_year / n) - 1)


def turnover(weights: pd.DataFrame) -> float:
    """Average daily sum of absolute weight changes across assets."""
    delta = weights.diff().abs().sum(axis=1)
    return float(delta.mean())


def hit_rate(returns: pd.Series) -> float:
    nonzero = returns[returns != 0]
    if len(nonzero) == 0:
        return 0.0
    return float((nonzero > 0).mean())


def summarize(returns: pd.Series, weights: pd.DataFrame | None = None) -> dict:
    equity = (1 + returns).cumprod()
    out = {
        "annualized_return": annualized_return(returns),
        "annualized_vol": float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "hit_rate": hit_rate(returns),
    }
    if weights is not None:
        out["avg_daily_turnover"] = turnover(weights)
    return out
