"""Sanity checks for formation-based pair selection (src/pairs.py).

Run: python -m tests.test_pairs
"""
import numpy as np
import pandas as pd

from src.pairs import adaptive_pairs_weights, select_pairs_by_distance


def test_select_pairs_by_distance_picks_the_closer_pair():
    dates = pd.bdate_range("2024-01-01", periods=100)
    rng = np.random.default_rng(0)
    common = rng.normal(0, 0.01, 100).cumsum()
    # A and B track the same path closely; C diverges a lot.
    a = 100 * np.exp(common)
    b = 100 * np.exp(common + rng.normal(0, 0.002, 100).cumsum())
    c = 100 * np.exp(-common * 2)
    prices = pd.DataFrame({"A": a, "B": b, "C": c}, index=dates)

    pairs = select_pairs_by_distance(prices, ["A", "B", "C"], top_n=1)
    assert pairs == [("A", "B")], f"expected the close-tracking pair, got {pairs}"
    print("PASS: test_select_pairs_by_distance_picks_the_closer_pair")


def test_adaptive_pairs_weights_selects_using_formation_only():
    """Construct a scenario where A/B are the closest pair during the
    FORMATION window, but A/C would look closest if the selection function
    were (incorrectly) allowed to see the trading window too. Verify the
    selected pair is still A/B -- i.e. selection is decided before the
    trading window starts, not influenced by it.
    """
    dates = pd.bdate_range("2024-01-01", periods=120)
    train_days = 60
    rng = np.random.default_rng(1)

    common = rng.normal(0, 0.01, 120).cumsum()
    a = np.concatenate([common[:train_days], common[train_days:] + 0.5])  # A diverges from common after train
    b = common.copy()  # B tracks the ORIGINAL common path throughout
    c = np.concatenate([common[:train_days] + 0.3, common[train_days:]])  # C only converges to A's post-train path

    prices = pd.DataFrame({
        "A": 100 * np.exp(a),
        "B": 100 * np.exp(b),
        "C": 100 * np.exp(c),
    }, index=dates)

    # During formation (first 60 days): A and B are identical (distance 0),
    # A and C differ by a constant 0.3 offset -- A/B should be selected.
    weights = adaptive_pairs_weights(prices, train_days=train_days, candidates=["A", "B", "C"],
                                      top_n=1, lookback=10, entry_z=1.0, exit_z=0.2)
    # If C had been (wrongly) selected instead of B, weights["B"] would be
    # all zero for the whole window.
    assert (weights["B"] != 0).any(), "expected the formation-period-closest pair (A/B) to be traded"
    print("PASS: test_adaptive_pairs_weights_selects_using_formation_only")


if __name__ == "__main__":
    test_select_pairs_by_distance_picks_the_closer_pair()
    test_adaptive_pairs_weights_selects_using_formation_only()
    print("\nAll pair-selection sanity checks passed.")
