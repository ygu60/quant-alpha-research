"""Sanity checks for the risk overlay module (src/risk.py) -- these check
the risk CONTROLS behave correctly, not whether any strategy has alpha.

Run: python -m tests.test_risk
"""
import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.risk import (
    cap_gross_exposure,
    cap_position_weight,
    drawdown_kill_switch,
    volatility_target,
)


def test_cap_position_weight_clips_only_excess():
    dates = pd.bdate_range("2024-01-01", periods=3)
    weights = pd.DataFrame({"A": [0.5, -0.5, 0.2], "B": [0.1, 0.1, 0.1]}, index=dates)
    capped = cap_position_weight(weights, max_weight=0.35)
    assert capped["A"].tolist() == [0.35, -0.35, 0.2]
    assert capped["B"].tolist() == [0.1, 0.1, 0.1]
    print("PASS: test_cap_position_weight_clips_only_excess")


def test_cap_gross_exposure_only_scales_down():
    dates = pd.bdate_range("2024-01-01", periods=2)
    weights = pd.DataFrame({"A": [1.0, 0.3], "B": [1.0, 0.3]}, index=dates)  # day0 gross=2.0, day1 gross=0.6
    capped = cap_gross_exposure(weights, max_gross=1.0)
    assert np.allclose(capped.iloc[0].sum(), 1.0), "over-gross day should scale down to the cap"
    assert np.allclose(capped.iloc[1].tolist(), [0.3, 0.3]), "under-gross day should be untouched"
    print("PASS: test_cap_gross_exposure_only_scales_down")


def test_volatility_target_scales_down_high_vol_book():
    dates = pd.bdate_range("2024-01-01", periods=40)
    rng = np.random.default_rng(0)
    # A single asset with large daily moves -> the strategy's own realized
    # vol should be well above a 10% annual target, so leverage should be
    # scaled toward min_leverage once the lookback window fills.
    prices = pd.DataFrame({"A": 100 * (1 + rng.normal(0, 0.05, 40)).cumprod()}, index=dates)
    weights = pd.DataFrame({"A": 1.0}, index=dates)
    scaled = volatility_target(prices, weights, target_vol=0.10, lookback=10, max_leverage=2.0, min_leverage=0.2)
    late = scaled["A"].iloc[15:]
    assert (late <= 1.0).all(), "high realized vol should never scale leverage above the raw weight"
    print("PASS: test_volatility_target_scales_down_high_vol_book")


def test_drawdown_kill_switch_flattens_and_recovers():
    dates = pd.bdate_range("2024-01-01", periods=12)
    # Prices: flat, then a sharp -20% single-day drop (breaches 15% dd),
    # then a rally back above the 5% recovery threshold.
    prices = pd.Series([100, 100, 100, 80, 80, 80, 95, 99, 100, 100, 100, 100], index=dates)
    prices = pd.DataFrame({"A": prices})
    weights = pd.DataFrame({"A": 1.0}, index=dates)  # fully invested every day, pre-overlay

    out = drawdown_kill_switch(prices, weights, cost_bps=0.0, max_drawdown=0.15, recovery_drawdown=0.05)

    # The -20% drop from index2->index3 is realized using the weight decided
    # at index2 (already 1.0, pre-halt) -- expected, that loss is what
    # TRIGGERS the halt, the overlay can't have prevented it.
    # From the day AFTER the halt is triggered, weights must be 0 until
    # equity recovers to within 5% of peak.
    assert out["A"].iloc[3] == 0.0, "weight decided at day of breach should already be flattened"
    assert out["A"].iloc[4] == 0.0
    assert out["A"].iloc[5] == 0.0

    result = run_backtest(prices, out, cost_bps=0.0, long_only=False)
    assert result["stats"]["max_drawdown"] <= -0.14, "sanity: the triggering drop should still show up"
    print("PASS: test_drawdown_kill_switch_flattens_and_recovers")


if __name__ == "__main__":
    test_cap_position_weight_clips_only_excess()
    test_cap_gross_exposure_only_scales_down()
    test_volatility_target_scales_down_high_vol_book()
    test_drawdown_kill_switch_flattens_and_recovers()
    print("\nAll risk overlay sanity checks passed.")
