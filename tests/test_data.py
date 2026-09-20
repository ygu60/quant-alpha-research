"""Sanity checks for the data-hygiene layer (src/data.py) -- these use
hand-constructed panels, not real network calls, so they run offline and
fast. The point is to verify align_universe's drop/trim logic, since a bug
there would silently change which names a backtest sees.

Run: python -m tests.test_data
"""
import unittest.mock as mock
import urllib.error
import urllib.request as _ur

import numpy as np
import pandas as pd

from src.data import DataFetchError, align_universe, fetch_yahoo_history


def _panel(n_days=600):
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rng = np.random.default_rng(0)
    base = 100 * (1 + rng.normal(0, 0.01, n_days)).cumprod()
    return dates, base


def test_align_universe_drops_short_history():
    dates, base = _panel()
    prices = pd.DataFrame({"A": base, "B": base * 1.1})
    prices.index = dates
    # C only has 100 observations -- below the min_obs=500 default
    prices["C"] = np.nan
    prices.loc[dates[:100], "C"] = base[:100]

    clean, report = align_universe(prices, min_obs=500)
    assert "C" in report["dropped_short_history"]
    assert "C" not in clean.columns
    assert set(clean.columns) == {"A", "B"}
    print("PASS: test_align_universe_drops_short_history")


def test_align_universe_drops_stale_names():
    dates, base = _panel()
    prices = pd.DataFrame({"A": base}, index=dates)
    # B has 550 valid observations but stops 60 days before the panel end --
    # simulates a delisted/stopped-trading ticker.
    b = base.copy()
    b[-60:] = np.nan
    prices["B"] = b

    clean, report = align_universe(prices, min_obs=500, max_staleness_days=10)
    assert any(name == "B" for name, _ in report["dropped_stale"])
    assert "B" not in clean.columns
    print("PASS: test_align_universe_drops_stale_names")


def test_align_universe_trims_start_to_high_coverage():
    dates, base = _panel()
    prices = pd.DataFrame({"A": base}, index=dates)
    # B only starts halfway through (late IPO) but has enough total
    # observations to survive the min_obs check on its own.
    b = base.copy()
    b[:300] = np.nan
    prices["B"] = b

    clean, report = align_universe(prices, min_obs=250, min_coverage=0.99)
    assert clean.index.min() >= dates[300]
    assert clean.notna().all().all(), "trimmed panel should have full coverage from its new start"
    print("PASS: test_align_universe_trims_start_to_high_coverage")


def test_fetch_yahoo_history_404_is_permanent_no_retry_storm():
    """404 (symbol not found) must fail fast, not burn through max_retries
    with sleep backoff -- that's the right behavior for a permanent error
    but would silently slow down a multi-ticker fetch_universe() call if
    404s were retried like transient network errors.
    """
    with mock.patch.object(_ur, "urlopen", side_effect=urllib.error.HTTPError("url", 404, "Not Found", {}, None)) as m:
        try:
            fetch_yahoo_history("NOTREAL", "2020-01-01", "2020-02-01", use_cache=False)
            assert False, "expected DataFetchError"
        except DataFetchError:
            pass
        assert m.call_count == 1, f"404 should not be retried, got {m.call_count} attempts"
    print("PASS: test_fetch_yahoo_history_404_is_permanent_no_retry_storm")


if __name__ == "__main__":
    test_align_universe_drops_short_history()
    test_align_universe_drops_stale_names()
    test_align_universe_trims_start_to_high_coverage()
    test_fetch_yahoo_history_404_is_permanent_no_retry_storm()
    print("\nAll data-layer sanity checks passed.")
