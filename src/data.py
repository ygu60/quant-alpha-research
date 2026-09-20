"""
Data layer for the research pipeline.

Design goal: keep strategy/backtest code completely agnostic to where prices
come from. Two sources are supported today:

1. load_csv(path)      -- real historical OHLCV data you supply (e.g. a CSV
                           exported from Yahoo Finance, Stooq, your broker, or
                           a vendor like Polygon/Alpaca/Tiingo).
2. simulate_universe()  -- synthetic multi-asset price paths used ONLY to
                           validate that the backtest math is correct
                           (Sharpe/drawdown/turnover calculations, no
                           look-ahead bias, walk-forward splitting, etc).
                           Results on synthetic data say NOTHING about real
                           alpha -- they only prove the plumbing works.

Why synthetic data exists at all: this sandbox currently has no outbound
network access to market-data providers (Yahoo/Stooq/FRED endpoints all come
back empty or blocked). See RESEARCH_MEMO.md for how to plug in real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_csv(path: str, date_col: str = "Date", price_col: str = "Close") -> pd.Series:
    """Load a single-asset price series from a CSV file.

    Expects a column of dates and a column of prices (e.g. standard
    Yahoo/Stooq export columns: Date, Open, High, Low, Close, Volume).
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    s = df[price_col].astype(float)
    s.name = price_col
    return s


def load_csv_universe(paths: dict[str, str], date_col: str = "Date", price_col: str = "Close") -> pd.DataFrame:
    """Load multiple single-asset CSVs into one aligned price DataFrame.

    `paths` maps ticker -> csv file path.
    """
    series = {ticker: load_csv(p, date_col, price_col) for ticker, p in paths.items()}
    return pd.DataFrame(series).dropna(how="all").sort_index()


def simulate_universe(
    n_assets: int = 8,
    n_days: int = 1500,
    seed: int = 7,
    mean_reversion_strength: float = 0.06,
    drift: float = 0.00025,
    daily_vol: float = 0.02,
    common_factor_weight: float = 0.35,
) -> pd.DataFrame:
    """Generate a synthetic universe of small-cap-like price series.

    Each asset = random walk with (a) a shared market factor, (b) idiosyncratic
    drift, and (c) a mean-reverting component injected on top so short-horizon
    reversal/pairs strategies have *something real* to find. This is a
    deliberately simple model -- it exists to sanity-check the pipeline, not
    to represent any specific market.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_days)

    market_factor = rng.normal(0, daily_vol, n_days)
    tickers = [f"SIM{i+1}" for i in range(n_assets)]
    log_prices = {}

    for i, tkr in enumerate(tickers):
        idio = rng.normal(drift, daily_vol, n_days)
        raw_returns = common_factor_weight * market_factor + (1 - common_factor_weight) * idio

        # Inject a short-horizon mean-reverting overlay: today's overlay
        # partially cancels yesterday's overlay shock, e.g. Lehmann (1990)
        # style short-term reversal.
        overlay = np.zeros(n_days)
        shocks = rng.normal(0, daily_vol * 0.6, n_days)
        for t in range(1, n_days):
            overlay[t] = -mean_reversion_strength * overlay[t - 1] + shocks[t]

        total_returns = raw_returns + overlay
        log_prices[tkr] = np.cumsum(total_returns)

    prices = pd.DataFrame(log_prices, index=dates)
    prices = 50 * np.exp(prices)  # start around $50/share
    return prices
