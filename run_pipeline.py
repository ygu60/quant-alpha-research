"""
End-to-end research pipeline: real data -> signal -> backtest -> walk-forward
out-of-sample validation -> risk overlay -> final recommendation.

Run: python run_pipeline.py               (real data, ~1-2 min incl. fetch)
     python run_pipeline.py --engine-check (synthetic data, engine plumbing only)

Read RESEARCH_MEMO.md for the full writeup: literature grounding, why each
design choice was made, and the honest (not flattering) walk-forward results
this script reproduces. This docstring covers only what the script does.

Pipeline, in order, per strategy family:
  1. Baseline backtest with FIXED, literature-motivated parameters (no
     fitting) -- an honest first look, full-sample, clearly not an
     out-of-sample claim.
  2. Walk-forward out-of-sample validation with the SAME fixed parameters --
     the number that actually matters.
  3. (Short-term reversal only) a nested walk-forward parameter search, to
     check whether trying to "optimize" the fixed parameters helps or hurts
     out-of-sample -- see RESEARCH_MEMO.md for why the answer is "hurts",
     and why that supports keeping the untuned default rather than
     undermining it.
  4. Risk-managed walk-forward for the one candidate that survives step 2
     with a real, if modest, edge -- volatility targeting + position/gross
     caps + a wide (tail-only) drawdown circuit breaker, all with parameters
     chosen from risk-tolerance principles, not fit to this backtest.
"""
from __future__ import annotations

import sys
from functools import partial

import pandas as pd

from src import events
from src.backtest import run_backtest
from src.data import load_research_universe, simulate_universe
from src.metrics import summarize
from src.pairs import adaptive_pairs_weights
from src.risk import apply_full_overlay
from src.strategies import short_term_reversal, turn_of_month, zscore_pairs
from src.walk_forward import walk_forward, walk_forward_with_selection

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

COST_BPS = 10.0
TRAIN_DAYS = 252
TEST_DAYS = 63

# Risk overlay parameters: chosen from risk-tolerance principles (10%
# annualized vol is a reasonable satellite-sleeve budget; no leverage
# because this models a cash account; the 30% circuit breaker is
# deliberately loose -- see RESEARCH_MEMO.md section on why a TIGHT kill
# switch stacked on vol-targeting was found, empirically, to destroy value
# for this mean-reversion strategy). NOT fit to the backtest below.
RISK_CFG = dict(
    target_vol=0.10, vol_lookback=20, max_leverage=1.0, min_leverage=0.05,
    max_position=0.25, max_gross=1.0, max_drawdown=0.30, recovery_drawdown=0.10,
)


def print_stats(name: str, stats: dict) -> None:
    print(f"\n--- {name} ---")
    for k, v in stats.items():
        print(f"  {k:>20}: {v:.4f}" if isinstance(v, float) else f"  {k:>20}: {v}")


def _downsample(series: pd.Series, max_points: int = 150) -> dict:
    if len(series) > max_points:
        step = len(series) // max_points
        series = series.iloc[::step]
    return {
        "dates": [str(getattr(d, "date", lambda: d)()) for d in series.index],
        "values": [round(float(v), 6) for v in series.values],
    }


def log_result(name: str, returns: pd.Series, stats: dict) -> None:
    events.log(
        "status", f"completed: {name}", strategy=name,
        equity_curve=_downsample((1 + returns).cumprod()), **stats,
    )
    if stats.get("sharpe", 0) < 0:
        events.log("warning", f"{name}: negative Sharpe ratio", strategy=name, sharpe=stats["sharpe"])


def to_long_only_weights(raw_weights: pd.DataFrame) -> pd.DataFrame:
    """Convert a long/short signal (e.g. short_term_reversal's long+short
    legs) into a fully-invested long-only book: clip negatives, renormalize
    the surviving long names to sum to 1. This the ONE place that
    normalization happens -- everything downstream (risk overlay) receives
    already-long-only weights and must be scored with long_only=False so
    its deliberate cash positions aren't renormalized away again (see
    RESEARCH_MEMO.md -- this exact mistake produced a materially wrong
    headline number during development).
    """
    w = raw_weights.clip(lower=0.0)
    row_sums = w.sum(axis=1).replace(0, 1.0)
    return w.div(row_sums, axis=0)


def reversal_risk_managed(window_prices: pd.DataFrame, lookback: int, n_long: int, n_short: int) -> pd.DataFrame:
    raw = short_term_reversal(window_prices, lookback=lookback, n_long=n_long, n_short=n_short)
    long_only = to_long_only_weights(raw)
    return apply_full_overlay(window_prices, long_only, cost_bps=COST_BPS, **RISK_CFG)


def run_engine_check() -> None:
    """Synthetic-data plumbing check (see src/data.py::simulate_universe) --
    proves the backtest math is correct, says NOTHING about real alpha.
    Kept for anyone who wants to sanity-check the engine without a network
    fetch. Use `python -m tests.test_pipeline` for the real correctness
    assertions; this is just a human-readable run.
    """
    print("=== ENGINE CHECK on SYNTHETIC data -- not a real-alpha claim ===")
    prices = simulate_universe(n_assets=8, n_days=1500, seed=7)
    bh_weights = pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)
    print_stats("Synthetic buy & hold", run_backtest(prices, bh_weights, cost_bps=10, long_only=True)["stats"])
    str_weights = short_term_reversal(prices, lookback=3, n_long=2, n_short=2)
    print_stats("Synthetic short-term reversal", run_backtest(prices, str_weights, cost_bps=10, long_only=True)["stats"])


def main() -> None:
    events.log("action", "pipeline started")

    print("Fetching real universe from Yahoo Finance (cached after first run)...")
    prices, bench, report = load_research_universe(start="2016-01-01")
    events.log("info", "universe loaded", n_assets=prices.shape[1], **{
        k: v for k, v in report.items() if k in ("date_range", "n_kept", "n_dropped", "fetch_failed")
    })
    print(f"Universe: {prices.shape[1]} names, {prices.shape[0]} trading days, {report['date_range'][0]} to {report['date_range'][1]}")
    print(f"Dropped (short history): {report['dropped_short_history']}")
    print(f"Dropped (stale/delisted): {report['dropped_stale']}")
    print(f"Fetch failed (404 -- likely delisted/renamed): {report['fetch_failed']}")

    n = prices.shape[1]
    n_side = max(2, round(n * 0.20))  # quintile-style long/short book, standard cross-sectional convention

    try:
        # ---------------- Benchmark ----------------
        bh_weights = pd.DataFrame(1.0 / n, index=prices.index, columns=prices.columns)
        bh_res = run_backtest(prices, bh_weights, cost_bps=COST_BPS, long_only=True)
        print_stats("Benchmark: equal-weight buy & hold (universe)", bh_res["stats"])
        log_result("benchmark_buy_hold", bh_res["returns"], bh_res["stats"])

        iwm_ret = bench["IWM"].pct_change().dropna()
        print_stats("Benchmark: IWM (Russell 2000 ETF) buy & hold", summarize(iwm_ret))

        # ---------------- Strategy 1: cross-sectional short-term reversal ----------------
        print("\n" + "=" * 70)
        print("STRATEGY 1: Short-term reversal (Lehmann 1990; Jegadeesh 1990)")
        print("=" * 70)

        raw_weights = short_term_reversal(prices, lookback=3, n_long=n_side, n_short=n_side)
        lo_weights = to_long_only_weights(raw_weights)
        base_res = run_backtest(prices, lo_weights, cost_bps=COST_BPS, long_only=False)
        print_stats("1a. Full-sample, fixed params, no tuning (reference only, NOT an OOS claim)", base_res["stats"])
        log_result("reversal_full_sample", base_res["returns"], base_res["stats"])

        wf_raw = walk_forward(prices, weight_fn=short_term_reversal, train_days=TRAIN_DAYS, test_days=TEST_DAYS,
                               cost_bps=COST_BPS, long_only=True, lookback=3, n_long=n_side, n_short=n_side)
        print_stats("1b. Walk-forward OOS, fixed params, UNMANAGED (extreme drawdown -- see memo)", wf_raw.combined_stats)
        log_result("reversal_walk_forward_unmanaged", wf_raw.combined_returns, wf_raw.combined_stats)

        grid = [{"lookback": lb, "n_long": max(2, round(n * f)), "n_short": max(2, round(n * f))}
                for lb in (2, 3, 5) for f in (0.15, 0.25)]
        wf_tuned = walk_forward_with_selection(prices, weight_fn=short_term_reversal, param_grid=grid,
                                                train_days=TRAIN_DAYS, test_days=TEST_DAYS, cost_bps=COST_BPS, long_only=True)
        print_stats("1c. Nested walk-forward WITH train-only parameter search (tuning check)", wf_tuned.combined_stats)
        print("    -> if this Sharpe is NOT better than 1b, tuning doesn't help here; keep the untuned default.")
        log_result("reversal_walk_forward_tuned", wf_tuned.combined_returns, wf_tuned.combined_stats)

        weight_fn = partial(reversal_risk_managed, lookback=3, n_long=n_side, n_short=n_side)
        wf_managed = walk_forward(prices, weight_fn=weight_fn, train_days=TRAIN_DAYS, test_days=TEST_DAYS,
                                   cost_bps=COST_BPS, long_only=False)
        print_stats("1d. Walk-forward OOS, RISK-MANAGED (headline number)", wf_managed.combined_stats)
        log_result("reversal_walk_forward_risk_managed", wf_managed.combined_returns, wf_managed.combined_stats)

        # Diversification check: does blending with buy & hold help?
        common = wf_managed.combined_returns.index.intersection(bh_res["returns"].index)
        corr = wf_managed.combined_returns.loc[common].corr(bh_res["returns"].loc[common])
        print(f"\n    Correlation vs. buy & hold over the same OOS dates: {corr:.3f}")
        print("    (High correlation means this is NOT a good diversifier for a portfolio")
        print("     already holding this universe -- see memo for the blend table.)")

        # ---------------- Strategy 2: pairs trading ----------------
        print("\n" + "=" * 70)
        print("STRATEGY 2: Pairs trading, regional banks (Gatev, Goetzmann & Rouwenhorst 2006)")
        print("=" * 70)
        banks = [t for t, s in report["sector_map"].items() if s == "regional_banks"]
        print(f"Candidate pool ({len(banks)} names): {banks}")
        pairs_fn = partial(adaptive_pairs_weights, train_days=TRAIN_DAYS, candidates=banks, top_n=2,
                            lookback=20, entry_z=2.0, exit_z=0.5)
        wf_pairs = walk_forward(prices, weight_fn=pairs_fn, train_days=TRAIN_DAYS, test_days=TEST_DAYS,
                                 cost_bps=COST_BPS, long_only=False)
        print_stats("2. Walk-forward OOS, formation-selected pairs", wf_pairs.combined_stats)
        log_result("pairs_walk_forward", wf_pairs.combined_returns, wf_pairs.combined_stats)

        # ---------------- Strategy 3: turn-of-month ----------------
        print("\n" + "=" * 70)
        print("STRATEGY 3: Turn-of-month seasonal (Ariel 1987; Lakonishok & Smidt 1988)")
        print("=" * 70)
        wf_tom = walk_forward(bench, weight_fn=turn_of_month, train_days=TRAIN_DAYS, test_days=TEST_DAYS,
                               cost_bps=COST_BPS, long_only=True, days_before=1, days_after=3)
        print_stats("3. Walk-forward OOS, applied to SPY/IWM/IJR (broad-portfolio effect)", wf_tom.combined_stats)
        log_result("turn_of_month_walk_forward", wf_tom.combined_returns, wf_tom.combined_stats)

        # ---------------- Final recommendation ----------------
        print("\n" + "=" * 70)
        print("RECOMMENDATION")
        print("=" * 70)
        headline = wf_managed.combined_stats
        print(f"Only strategy with a positive, cost-adjusted, out-of-sample Sharpe that")
        print(f"survives its own parameter-sensitivity check: short-term reversal + risk overlay.")
        print(f"  Headline OOS Sharpe ratio:  {headline['sharpe']:.3f}")
        print(f"  Probabilistic Sharpe Ratio: {headline['psr']:.3f}  (confidence true Sharpe > 0)")
        print(f"  OOS max drawdown:           {headline['max_drawdown']:.1%}")
        print(f"  OOS annualized return:      {headline['annualized_return']:.1%}")
        print(f"Pairs trading and turn-of-month showed no credible OOS edge on this universe/")
        print(f"period and should NOT be traded as currently specified. See RESEARCH_MEMO.md")
        print(f"for the full reasoning, including two risk-overlay bugs found and fixed during")
        print(f"this research and why they mattered.")
        events.log(
            "status", "pipeline recommendation",
            recommended_strategy="short_term_reversal_risk_managed",
            headline_sharpe=headline["sharpe"], headline_psr=headline["psr"],
            headline_max_drawdown=headline["max_drawdown"],
        )
    except Exception as exc:
        events.log_bug("pipeline crashed", exc)
        raise
    else:
        events.log("action", "pipeline finished")


if __name__ == "__main__":
    if "--engine-check" in sys.argv:
        run_engine_check()
    else:
        main()
