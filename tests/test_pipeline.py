"""
Sanity checks for the backtest engine itself -- these are NOT about whether
any strategy has alpha, they're about whether the plumbing can be trusted.

Run: python -m tests.test_pipeline   (from the quant-alpha-research/ folder)
"""
import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.metrics import sharpe_ratio, sortino_ratio


def test_no_lookahead():
    """A signal computed using day t's close must not affect day t's own
    return. It should only show up in the return from day t to day t+1.
    """
    dates = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame({"A": [100, 100, 100, 100, 200, 200]}, index=dates)

    # Weight goes to 1.0 exactly on the day of the price jump (day index 3->4).
    weights = pd.DataFrame({"A": [0, 0, 0, 1.0, 0, 0]}, index=dates)

    result = run_backtest(prices, weights, cost_bps=0, long_only=False)
    returns = result["returns"]

    # The 100% single-day jump happens between index 3 and 4. Because weights
    # are shifted forward one day before being applied, the weight set AT
    # index 3 should be applied to the index3->index4 return -- i.e. the
    # strategy SHOULD capture this jump. If it captured the jump one day
    # early (index2->index3, which is flat) or missed it, that's a bug.
    jump_return = returns.loc[dates[4]]
    assert abs(jump_return - 1.0) < 1e-9, f"expected to capture the 100% jump, got {jump_return}"

    day_before = returns.loc[dates[3]]
    assert abs(day_before - 0.0) < 1e-9, f"leaked future return into day 3: {day_before}"
    print("PASS: test_no_lookahead")


def test_sharpe_matches_manual_calculation():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.01, 500))
    expected = (rets.mean() / rets.std(ddof=1)) * np.sqrt(252)
    got = sharpe_ratio(rets)
    assert abs(expected - got) < 1e-9, f"{expected} vs {got}"
    print("PASS: test_sharpe_matches_manual_calculation")


def test_sortino_matches_empyrical():
    """Sortino & van der Meer's downside deviation is a target semi-
    deviation over the FULL sample (upside days contribute a zero, they are
    not excluded from the count) -- cross-checked against the standard
    `empyrical` library rather than just a hand-rolled formula, since an
    earlier version of this function used a non-standard definition
    (std over only the negative-return subset) that silently inflated the
    reported ratio. See src/metrics.py::sortino_ratio docstring.
    """
    import empyrical as ep
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.0006, 0.012, 800))
    ours = sortino_ratio(rets)
    theirs = ep.sortino_ratio(rets, required_return=0, annualization=252)
    assert abs(ours - theirs) < 1e-9, f"{ours} vs empyrical's {theirs}"
    print("PASS: test_sortino_matches_empyrical")


def test_costs_reduce_returns():
    dates = pd.bdate_range("2024-01-01", periods=10)
    rng = np.random.default_rng(1)
    prices = pd.DataFrame({"A": 100 * (1 + rng.normal(0, 0.01, 10)).cumprod()}, index=dates)
    weights = pd.DataFrame({"A": rng.choice([0.0, 1.0], 10)}, index=dates)

    free = run_backtest(prices, weights, cost_bps=0, long_only=False)
    costly = run_backtest(prices, weights, cost_bps=50, long_only=False)

    assert costly["returns"].sum() <= free["returns"].sum(), "transaction costs should never help returns"
    print("PASS: test_costs_reduce_returns")


def test_long_only_clips_negative_weights():
    dates = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"A": [100, 101, 99, 102, 103], "B": [50, 49, 51, 50, 52]}, index=dates)
    weights = pd.DataFrame({"A": [1.0, -1.0, 0.5, -0.5, 1.0], "B": [-1.0, 1.0, 0.5, 1.5, 0.0]}, index=dates)

    result = run_backtest(prices, weights, cost_bps=0, long_only=True)
    applied = result["weights"]
    assert (applied >= 0).all().all(), "long_only leaked a negative weight through"
    row_sums = applied.sum(axis=1)
    nonzero = row_sums[row_sums > 1e-9]
    assert np.allclose(nonzero, 1.0), f"long-only weights should renormalize to 1.0, got {nonzero.tolist()}"
    print("PASS: test_long_only_clips_negative_weights")


if __name__ == "__main__":
    test_no_lookahead()
    test_sharpe_matches_manual_calculation()
    test_sortino_matches_empyrical()
    test_costs_reduce_returns()
    test_long_only_clips_negative_weights()
    print("\nAll sanity checks passed.")
