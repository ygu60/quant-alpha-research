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
    """Sortino & van der Meer's (1991) downside deviation is a target
    semi-deviation over the FULL sample -- upside days contribute a zero,
    they are not excluded from the count:

        DD = sqrt( (1/N) * sum_t min(r_t - MAR, 0)^2 )

    An earlier version of this function computed std(ddof=1) over only the
    negative-excess-return subset, i.e. divided by (n_negative_days - 1)
    instead of N. That's a different, non-standard statistic, and it
    systematically UNDERSTATES downside deviation (inflating the reported
    Sortino ratio) whenever the return series has more up-days than
    down-days, which is the typical case for a positive-drift book. Caught
    by cross-checking against `empyrical.sortino_ratio` on the same series,
    which disagreed by no small margin.
    """
    excess = returns - rf / periods_per_year
    downside_sq = np.minimum(excess, 0.0) ** 2
    dd = np.sqrt(downside_sq.mean())
    if dd == 0 or np.isnan(dd):
        return 0.0
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


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


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    benchmark_sharpe: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012,
    "The Sharpe Ratio Efficient Frontier").

    A raw Sharpe ratio from a finite, possibly skewed/fat-tailed return
    series overstates confidence in the true (population) Sharpe. PSR asks:
    given the OBSERVED Sharpe, sample size, skew, and kurtosis, what is the
    probability the TRUE Sharpe exceeds `benchmark_sharpe` (default 0)?

    PSR = Phi( (SR_hat - SR*) * sqrt(n-1) / sqrt(1 - skew*SR_hat + (kurt-1)/4 * SR_hat^2) )

    where SR_hat is the per-period (not annualized) Sharpe, skew/kurt are the
    sample skewness/(non-excess) kurtosis of returns, and Phi is the
    standard normal CDF. Values well below ~0.95 mean "this Sharpe ratio
    could easily be noise" even if the annualized number looks attractive --
    exactly the failure mode a small-sample backtest is prone to and a
    trustworthy writeup should report, not hide.
    """
    from scipy.stats import norm

    n = len(returns)
    if n < 3:
        return float("nan")
    sr_hat = sharpe_ratio(returns, periods_per_year=1)  # per-period, not annualized
    sr_star = benchmark_sharpe / np.sqrt(periods_per_year)
    skew = float(returns.skew())
    kurt = float(returns.kurtosis()) + 3.0  # pandas kurtosis() is excess kurtosis; formula wants raw

    denom = np.sqrt(max(1e-12, 1 - skew * sr_hat + ((kurt - 1) / 4) * sr_hat ** 2))
    z = (sr_hat - sr_star) * np.sqrt(n - 1) / denom
    return float(norm.cdf(z))


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
        "psr": probabilistic_sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "hit_rate": hit_rate(returns),
    }
    if weights is not None:
        out["avg_daily_turnover"] = turnover(weights)
    return out
