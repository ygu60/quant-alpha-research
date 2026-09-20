"""Sanity checks for src/features.py -- mainly that features are causal
(don't change if future prices change) and that forward_return computes
exactly the intended forward window.

Run: python -m tests.test_features
"""
import numpy as np
import pandas as pd

from src.features import compute_features, forward_return, stack_to_long


def test_features_are_causal():
    """Changing a price AFTER day t must not change any feature value AT
    day t. This is the feature-engineering equivalent of test_no_lookahead
    in tests/test_pipeline.py.
    """
    dates = pd.bdate_range("2024-01-01", periods=80)
    rng = np.random.default_rng(0)
    base = 100 * (1 + rng.normal(0, 0.01, 80)).cumprod()
    prices_a = pd.DataFrame({"X": base, "Y": base * 1.05}, index=dates)

    prices_b = prices_a.copy()
    shock_day = 60
    prices_b.iloc[shock_day + 1:] *= 3.0  # violently change everything AFTER day 60

    feats_a = compute_features(prices_a)
    feats_b = compute_features(prices_b)

    for name in feats_a:
        before = feats_a[name].iloc[:shock_day + 1]
        after = feats_b[name].iloc[:shock_day + 1]
        assert np.allclose(before.to_numpy(), after.to_numpy(), equal_nan=True), \
            f"feature '{name}' changed at/before day {shock_day} when only future prices changed -- look-ahead leak"
    print("PASS: test_features_are_causal")


def test_forward_return_matches_manual_calc():
    dates = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame({"A": [100, 102, 101, 105, 110, 108, 112, 115, 114, 120]}, index=dates)
    fwd = forward_return(prices, horizon=3)
    # fwd at day 0 should be (price[3]/price[0] - 1) = (105/100 - 1)
    assert abs(fwd["A"].iloc[0] - (105 / 100 - 1)) < 1e-12
    # last 3 rows must be NaN (no future data to compute the label)
    assert fwd["A"].iloc[-3:].isna().all()
    print("PASS: test_forward_return_matches_manual_calc")


def test_stack_to_long_shape_and_alignment():
    dates = pd.bdate_range("2024-01-01", periods=5)
    cols = ["A", "B"]
    f1 = pd.DataFrame(np.arange(10).reshape(5, 2), index=dates, columns=cols).astype(float)
    f2 = pd.DataFrame(np.arange(10, 20).reshape(5, 2), index=dates, columns=cols).astype(float)
    long_df = stack_to_long({"f1": f1, "f2": f2}, dates, cols)

    assert len(long_df) == 5 * 2
    assert list(long_df.columns) == ["f1", "f2"]
    # spot-check one specific (date, ticker) cell round-trips correctly
    assert long_df.loc[(dates[2], "B"), "f1"] == f1.loc[dates[2], "B"]
    print("PASS: test_stack_to_long_shape_and_alignment")


if __name__ == "__main__":
    test_features_are_causal()
    test_forward_return_matches_manual_calc()
    test_stack_to_long_shape_and_alignment()
    print("\nAll feature-engineering sanity checks passed.")
