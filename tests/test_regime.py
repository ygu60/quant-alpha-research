"""Sanity checks for src/regime.py.

Run: python -m tests.test_regime
"""
import numpy as np
import pandas as pd

from src.regime import (
    combined_rule_regime,
    fit_markov_params,
    high_vol_state_index,
    markov_regime_walk_forward,
    trend_regime,
    volatility_regime,
)


def test_trend_regime_is_causal_and_correct():
    dates = pd.bdate_range("2024-01-01", periods=250)
    # Flat at 100 for a long warm-up, then a sustained rally.
    prices = pd.Series(100.0, index=dates)
    prices.iloc[210:] = [100 + i for i in range(len(dates) - 210)]
    trend = trend_regime(prices, window=200)
    assert trend.iloc[-1] == True  # noqa: E712 -- price well above trailing MA by the end
    assert pd.isna(trend.iloc[50]) == False and trend.iloc[50] == False  # flat period: price == MA, not above
    print("PASS: test_trend_regime_is_causal_and_correct")


def test_volatility_regime_flags_a_vol_spike_as_high():
    dates = pd.bdate_range("2024-01-01", periods=600)
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.005, 600), index=dates)
    returns.iloc[550:570] = rng.normal(0, 0.05, 20)  # a late, sharp vol spike
    bucket = volatility_regime(returns, vol_window=10, percentile_window=500)
    assert bucket.iloc[575] == "high", f"expected 'high' during/after the vol spike, got {bucket.iloc[575]}"
    print("PASS: test_volatility_regime_flags_a_vol_spike_as_high")


def test_combined_rule_regime_shape():
    dates = pd.bdate_range("2024-01-01", periods=600)
    rng = np.random.default_rng(1)
    prices = pd.Series(100 * (1 + rng.normal(0.0003, 0.01, 600)).cumprod(), index=dates)
    combined = combined_rule_regime(prices)
    valid = combined.dropna()
    assert len(valid) > 0
    assert set(valid.unique()).issubset({"up_low", "up_mid", "up_high", "down_low", "down_mid", "down_high"})
    print("PASS: test_combined_rule_regime_shape")


def test_high_vol_state_index_picks_larger_variance():
    params = pd.Series({"sigma2[0]": 0.5, "sigma2[1]": 2.0, "const[0]": 0.01, "const[1]": 0.0,
                         "p[0->0]": 0.9, "p[1->0]": 0.1})
    assert high_vol_state_index(params) == 1
    params2 = pd.Series({"sigma2[0]": 3.0, "sigma2[1]": 0.2})
    assert high_vol_state_index(params2) == 0
    print("PASS: test_high_vol_state_index_picks_larger_variance")


def test_markov_walk_forward_runs_and_is_bounded_probability():
    dates = pd.bdate_range("2016-01-01", periods=700)
    rng = np.random.default_rng(2)
    # calm regime then a turbulent regime then calm again
    rets = np.concatenate([
        rng.normal(0.0003, 0.005, 300),
        rng.normal(-0.001, 0.03, 100),
        rng.normal(0.0003, 0.005, 300),
    ])
    returns = pd.Series(rets, index=dates)
    result = markov_regime_walk_forward(returns, train_days=252, test_days=63, k_regimes=2)
    assert len(result) > 0
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 1).all(), "filtered probabilities must be in [0, 1]"
    print("PASS: test_markov_walk_forward_runs_and_is_bounded_probability")


if __name__ == "__main__":
    test_trend_regime_is_causal_and_correct()
    test_volatility_regime_flags_a_vol_spike_as_high()
    test_combined_rule_regime_shape()
    test_high_vol_state_index_picks_larger_variance()
    test_markov_walk_forward_runs_and_is_bounded_probability()
    print("\nAll regime-detection sanity checks passed.")
