"""
End-to-end demo of the research pipeline: data -> signal -> backtest ->
walk-forward validation -> report.

Run: python run_pipeline.py

IMPORTANT: this script runs on SYNTHETIC data (src/data.py::simulate_universe)
because this environment has no live market-data access. The Sharpe ratios
below validate that the pipeline's math is correct (no look-ahead, costs
applied, walk-forward properly out-of-sample) -- they say nothing about real
alpha. Swap in `load_csv_universe(...)` with real data before drawing any
conclusion about actual trading edge. See RESEARCH_MEMO.md.
"""
from __future__ import annotations

import pandas as pd

from src.data import simulate_universe
from src.strategies import short_term_reversal, zscore_pairs, turn_of_month
from src.backtest import run_backtest
from src.walk_forward import walk_forward

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


def print_stats(name: str, stats: dict) -> None:
    print(f"\n--- {name} ---")
    for k, v in stats.items():
        print(f"  {k:>20}: {v:.4f}" if isinstance(v, float) else f"  {k:>20}: {v}")


def main() -> None:
    prices = simulate_universe(n_assets=8, n_days=1500, seed=7)
    print(f"Simulated universe: {prices.shape[1]} assets, {prices.shape[0]} trading days "
          f"({prices.index[0].date()} to {prices.index[-1].date()})")

    # 1. Buy-and-hold benchmark (equal weight, no rebalancing cost after day 1)
    bh_weights = pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)
    bh = run_backtest(prices, bh_weights, cost_bps=10, long_only=True)
    print_stats("Benchmark: equal-weight buy & hold", bh["stats"])

    # 2. Short-term reversal (long-only, since a small cash account
    #    typically can't short individual equities without margin)
    str_weights = short_term_reversal(prices, lookback=3, n_long=2, n_short=2)
    strat1 = run_backtest(prices, str_weights, cost_bps=10, long_only=True)
    print_stats("Strategy 1: short-term reversal (long-only)", strat1["stats"])

    # Also show the long/short (market-neutral) version for comparison
    strat1_ls = run_backtest(prices, str_weights, cost_bps=10, long_only=False)
    print_stats("Strategy 1b: short-term reversal (long/short, needs margin)", strat1_ls["stats"])

    # 3. Pairs trade on two of the simulated names
    pair_weights = zscore_pairs(prices, "SIM1", "SIM2", lookback=20, entry_z=1.5, exit_z=0.3)
    strat2 = run_backtest(prices, pair_weights, cost_bps=10, long_only=False)
    print_stats("Strategy 2: SIM1/SIM2 pairs z-score (long/short, needs margin)", strat2["stats"])

    # 4. Turn-of-month seasonal, long-only
    tom_weights = turn_of_month(prices, days_before=1, days_after=3)
    strat3 = run_backtest(prices, tom_weights, cost_bps=10, long_only=True)
    print_stats("Strategy 3: turn-of-month seasonal (long-only)", strat3["stats"])

    # 5. Walk-forward out-of-sample check on the strongest candidate
    print("\n=== Walk-forward validation: short-term reversal ===")
    wf = walk_forward(
        prices,
        weight_fn=short_term_reversal,
        train_days=252,
        test_days=63,
        cost_bps=10,
        long_only=True,
        lookback=3, n_long=2, n_short=2,
    )
    print(wf.window_stats.to_string(index=False))
    print_stats("Combined out-of-sample", wf.combined_stats)


if __name__ == "__main__":
    main()
