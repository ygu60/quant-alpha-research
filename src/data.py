"""
Data layer for the research pipeline.

Design goal: keep strategy/backtest code completely agnostic to where prices
come from. Sources supported:

1. fetch_yahoo_history() / fetch_universe() -- REAL daily OHLCV pulled from
   Yahoo Finance's public chart JSON endpoint
   (query1.finance.yahoo.com/v8/finance/chart/<ticker>). No API key needed,
   no third-party package needed (uses stdlib urllib). Split/dividend-adjusted
   close ("adjclose") is used so returns aren't contaminated by corporate
   actions. Responses are cached to disk (data/, gitignored) keyed by
   ticker+date range so re-running the pipeline doesn't re-hit the network.

   Note on provenance: an earlier pass in this project found Stooq's CSV
   export and FRED's series blocked/JS-walled from that sandbox and concluded
   no live data access existed at all. That turned out to be sandbox-specific
   -- this environment reaches Yahoo's chart endpoint fine (verified
   manually), so real history is used from here on. FRED-style JS-walled
   endpoints (Stooq) are still unreachable; Yahoo is the one source verified
   to work and is used exclusively below.

2. load_csv(path) / load_csv_universe() -- for CSVs you supply yourself
   (Yahoo/Stooq manual export, broker export, a paid vendor).

3. simulate_universe() -- synthetic multi-asset price paths used ONLY to
   validate that the backtest math is correct (Sharpe/drawdown/turnover
   calculations, no look-ahead bias, walk-forward splitting, etc). Results on
   synthetic data say NOTHING about real alpha -- they only prove the
   plumbing works. Kept for engine regression tests.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
_YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-pipeline/1.0"


class DataFetchError(RuntimeError):
    """Raised when a ticker's history can't be retrieved or parsed."""


def _cache_path(ticker: str, start: str, end: str) -> Path:
    safe = ticker.replace("/", "_").replace("^", "idx")
    return CACHE_DIR / f"{safe}_{start}_{end}.csv"


def fetch_yahoo_history(
    ticker: str,
    start: str,
    end: str,
    use_cache: bool = True,
    max_retries: int = 3,
    retry_delay_s: float = 1.5,
) -> pd.DataFrame:
    """Fetch one ticker's daily OHLCV (+ split/dividend-adjusted close) from
    Yahoo Finance's public chart endpoint.

    Parameters
    ----------
    ticker : e.g. "AAPL", or "^GSPC" / "^RUT" for indices.
    start, end : "YYYY-MM-DD" strings (inclusive-ish; Yahoo's `period1`/
        `period2` are unix seconds and treated as a bracket, not exact edges).
    use_cache : read/write a CSV cache under data/ so repeated pipeline runs
        don't re-hit the network for the same ticker+range.

    Returns a DataFrame indexed by date with columns
    [open, high, low, close, adjclose, volume]. Raises DataFetchError if the
    ticker can't be retrieved (e.g. delisted/typo/rate-limited after
    retries) -- callers should decide whether to drop the ticker or abort.
    """
    cache_file = _cache_path(ticker, start, end)
    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["date"], index_col="date")
        return df

    period1 = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    period2 = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    url = (
        _YAHOO_CHART_URL.format(ticker=ticker)
        + f"?period1={period1}&period2={period2}&interval=1d&events=div,splits"
    )

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.load(resp)
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise DataFetchError(f"{ticker}: symbol not found (HTTP 404)") from exc
            last_exc = exc
            time.sleep(retry_delay_s * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            time.sleep(retry_delay_s * (attempt + 1))
    else:
        raise DataFetchError(f"{ticker}: failed after {max_retries} attempts: {last_exc}")

    result = payload.get("chart", {}).get("result")
    if not result:
        err = payload.get("chart", {}).get("error")
        raise DataFetchError(f"{ticker}: no data returned (error={err})")

    r = result[0]
    timestamps = r.get("timestamp")
    if not timestamps:
        raise DataFetchError(f"{ticker}: empty timestamp series (likely delisted or invalid symbol)")

    quote = r["indicators"]["quote"][0]
    adjclose = r["indicators"].get("adjclose", [{}])[0].get("adjclose", quote["close"])

    dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
    df = pd.DataFrame(
        {
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "adjclose": adjclose,
            "volume": quote["volume"],
        },
        index=dates,
    )
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["adjclose"])

    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        df.to_csv(cache_file)

    return df


# Candidate universe for this study. Grouped by sector because the pairs
# strategy needs sector-matched candidates (economically linked names are
# the ones that plausibly cointegrate -- Gatev, Goetzmann & Rouwenhorst 2006
# form pairs within the same broad industry group for exactly this reason).
# Regional banks are the deepest bench here: correlated business model
# (net interest margin, regional credit cycle), semi-liquid, and small
# enough in trade-capacity terms that a big fund won't bother -- see
# RESEARCH_MEMO.md thesis section.
UNIVERSE_GROUPS: dict[str, list[str]] = {
    "regional_banks": ["WAL", "ZION", "CMA", "SNV", "FHN", "WBS", "ONB", "PNFP", "UMBF", "ABCB"],
    "industrials": ["WTS", "ITT", "KAI", "ATKR", "CIR"],
    "consumer_discretionary": ["BOOT", "FIVE", "WING", "CAKE", "SFM"],
    "healthcare": ["OMCL", "ICUI", "PDCO", "ADUS", "AMED"],
    "energy": ["MTDR", "SM", "PR", "CIVI", "MGY"],
}
BENCHMARK_TICKERS = ["SPY", "IWM", "IJR"]  # S&P 500 / Russell 2000 / S&P SmallCap 600


def align_universe(
    prices: pd.DataFrame,
    min_obs: int = 500,
    max_staleness_days: int = 10,
    min_coverage: float = 0.95,
) -> tuple[pd.DataFrame, dict]:
    """Data hygiene pass before any strategy sees the panel.

    Real multi-name panels have three problems synthetic data never has:
    late IPOs (leading NaNs), delistings/M&A (trailing NaNs -- a "stale"
    ticker whose last print is long before the panel's end), and short
    histories that are statistically useless. Silently forward-filling or
    ignoring these would let a delisted or too-short name quietly distort
    cross-sectional ranks and pair z-scores. Instead:

    1. Drop tickers with fewer than `min_obs` total observations.
    2. Drop tickers whose last valid observation is more than
       `max_staleness_days` before the panel's overall last date (delisted
       or otherwise stopped trading).
    3. Trim the START of the panel to the earliest date at which at least
       `min_coverage` fraction of the SURVIVING tickers have data, so
       early-panel cross-sectional stats aren't computed on 3 of 30 names.

    Returns (clean_prices, report) where report documents exactly what was
    dropped/trimmed and why -- this belongs in the research memo verbatim,
    not just in a log line, because silently shrinking a universe is a
    classic way backtests quietly become unrepresentative.
    """
    report: dict = {"dropped_short_history": [], "dropped_stale": [], "kept": [], "start_trimmed_to": None}

    panel_end = prices.index.max()
    keep_cols = []
    for col in prices.columns:
        s = prices[col].dropna()
        if len(s) < min_obs:
            report["dropped_short_history"].append(col)
            continue
        staleness = (panel_end - s.index.max()).days
        if staleness > max_staleness_days:
            report["dropped_stale"].append((col, str(s.index.max().date())))
            continue
        keep_cols.append(col)

    trimmed = prices[keep_cols]
    coverage = trimmed.notna().mean(axis=1)
    ok_dates = coverage[coverage >= min_coverage].index
    if len(ok_dates) == 0:
        raise DataFetchError("no date has sufficient coverage after alignment -- check min_coverage/min_obs")
    start = ok_dates.min()
    trimmed = trimmed.loc[start:]

    report["kept"] = keep_cols
    report["start_trimmed_to"] = str(start.date())
    report["n_kept"] = len(keep_cols)
    report["n_dropped"] = len(report["dropped_short_history"]) + len(report["dropped_stale"])
    return trimmed, report


def load_research_universe(
    start: str = "2016-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """One call that fetches the full study universe + benchmarks, aligns
    it, and returns (clean_universe_prices, benchmark_prices, data_report).

    `data_report` also records `sector_map` (surviving ticker -> sector) so
    downstream pair-selection code can restrict candidate pairs to
    economically-linked names instead of the whole cross-section.
    """
    end = end or datetime.now().strftime("%Y-%m-%d")
    all_tickers = [t for group in UNIVERSE_GROUPS.values() for t in group]

    prices, failed = fetch_universe(all_tickers, start, end, use_cache=use_cache)
    bench_prices, bench_failed = fetch_universe(BENCHMARK_TICKERS, start, end, use_cache=use_cache)

    clean, report = align_universe(prices)
    report["fetch_failed"] = failed
    report["benchmark_fetch_failed"] = bench_failed
    report["sector_map"] = {
        t: sector for sector, tickers in UNIVERSE_GROUPS.items() for t in tickers if t in clean.columns
    }
    report["date_range"] = [str(clean.index.min().date()), str(clean.index.max().date())]
    return clean, bench_prices, report


def load_research_volume(
    tickers: list[str],
    start: str,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch raw share volume for exactly the given tickers (typically the
    already-cleaned columns from `load_research_universe`) over the same
    date range, reusing the same per-ticker disk cache -- no extra network
    calls if `load_research_universe` already ran with the same date range.
    Not run through `align_universe` itself: it's meant to be reindexed
    against an already-aligned price panel by the caller (src/features.py
    needs price and volume on identical index/columns).
    """
    end = end or datetime.now().strftime("%Y-%m-%d")
    volume, failed = fetch_universe(tickers, start, end, use_cache=use_cache, field="volume")
    if failed:
        print(f"  [data] WARNING: volume fetch failed for {failed} (price data for these should already be absent)")
    return volume


def fetch_universe(
    tickers: list[str],
    start: str,
    end: str,
    use_cache: bool = True,
    sleep_between_s: float = 0.3,
    field: str = "adjclose",
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch one OHLCV `field` (default: split/dividend-adjusted close) for a
    list of tickers into one aligned DataFrame. Tickers that fail to fetch
    (delisted, typo'd, rate-limited) are dropped and returned separately
    rather than raising, so one bad ticker doesn't kill a 30-name universe
    fetch -- but the caller MUST look at (and report) the dropped list,
    since silently shrinking the universe changes the study. Cheap to call
    again with a different `field` (e.g. "volume") since the underlying
    per-ticker fetch is disk-cached regardless of which field is read from it.
    """
    series = {}
    failed = []
    for i, tkr in enumerate(tickers):
        try:
            hist = fetch_yahoo_history(tkr, start, end, use_cache=use_cache)
            series[tkr] = hist[field]
        except DataFetchError as exc:
            failed.append(tkr)
            print(f"  [data] WARNING dropping {tkr}: {exc}")
        if not use_cache or not _cache_path(tkr, start, end).exists():
            time.sleep(sleep_between_s)

    prices = pd.DataFrame(series).sort_index()
    return prices, failed


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
