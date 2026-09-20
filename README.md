# quant-alpha-research

Research/backtest pipeline for statistical trading strategies sized for a
small (single-thousands-dollar) account, built toward eventual live testing
via Robinhood's Agentic Trading MCP.

Start with `RESEARCH_MEMO.md` — it covers the literature grounding, the real
data universe and its data-quality quirks, walk-forward methodology, full
results per strategy (including several the pipeline does NOT recommend
trading), two risk-overlay bugs found and fixed during development, and
honest limitations to read before risking capital. `docs/` has a
textbook-style theory chapter per strategy (motivation, formal derivation,
falsifiable prediction) for anyone who wants to learn the underlying ideas,
not just the results.

**Headline result** (see memo §7.4 for full context): short-term
cross-sectional reversal, with a volatility-targeting risk overlay AND a
regime gate that sits out the lowest-volatility tertile of SPY's own
trailing history, walk-forward out-of-sample Sharpe ratio **0.44** (PSR
0.91, max drawdown -16.8%) over 23 real names, 2016–2026, net of
transaction costs, active 63% of days. The regime gate was cross-checked
across two different lookback windows AND an independent Markov-switching
model before being trusted — a narrower "high-vol only" version of the
same idea looked even better at one lookback window and then flipped to
strongly negative at another, and was rejected on exactly that basis (memo
§7.1b). Pairs trading and turn-of-month were also tested and are NOT
recommended as currently specified — no credible out-of-sample edge.

Two further follow-ups also found **no deployable edge**, documented as
negative results rather than discarded (memo §10): a non-semantic ML
signal search (Ridge/RandomForest/GradientBoosting on 15 price/volume
features, purged walk-forward, a permutation-test negative control) and a
small deep-learning parametric search (PyTorch MLP grid) both failed to
beat their own shuffled-label noise floor convincingly, and both went
*negative* once risk-managed with the same protocol as the headline
strategy — against SPY's 0.85 Sharpe / -33.7% drawdown over the identical
out-of-sample dates. See `plots/` for the equity-curve, drawdown,
rolling-Sharpe, feature-importance, and permutation-test charts that make
the comparisons immediate.

## Quickstart

```bash
pip install -r requirements.txt
python run_pipeline.py             # real data (fetches + caches from Yahoo Finance)
python run_pipeline.py --engine-check  # synthetic data, engine plumbing only, no network
python -m tests.test_pipeline      # backtest engine correctness checks
python -m tests.test_risk          # risk overlay correctness checks
python dashboard.py                # live terminal view of the most recent pipeline run
```

## Layout

- `src/data.py` — real historical data via Yahoo Finance's public chart
  endpoint (`fetch_yahoo_history`, `fetch_universe`), data-hygiene pass for
  delistings/short-history/ticker-reassignment (`align_universe`), the
  study universe definition (`load_research_universe`), plus a CSV loader
  and a synthetic generator kept for engine regression tests.
- `src/strategies.py` — short-term reversal, single-pair zscore mean
  reversion, turn-of-month.
- `src/pairs.py` — formation-period pair SELECTION (Gatev, Goetzmann &
  Rouwenhorst 2006 distance method) on top of `strategies.zscore_pairs`, so
  which pair to trade is chosen before the window it's evaluated on.
- `src/risk.py` — composable risk overlay: volatility targeting, per-name
  and gross exposure caps, and a drawdown kill switch.
- `src/backtest.py` — vectorized backtest with cost model and look-ahead
  protection (weights are shifted forward one day before being applied).
- `src/walk_forward.py` — rolling out-of-sample validation, plus nested
  (train-only) parameter search.
- `src/metrics.py` — Sharpe, Sortino, Probabilistic Sharpe Ratio, max
  drawdown, turnover, hit rate.
- `src/features.py` — non-semantic price/volume feature engineering for the
  ML signal search (causal by construction; see `tests/test_features.py`).
- `src/ml.py` — purged walk-forward ML training/prediction/portfolio
  construction, with a `shuffle_labels` negative control for leakage/noise
  checks (Lopez de Prado-style purging; see `tests/test_ml.py`).
- `src/dl.py` — a small PyTorch MLP wrapped with an sklearn-style fit/
  predict interface, so it plugs into the same purged walk-forward loop as
  every other model in `src/ml.py` (see `tests/test_dl.py`).
- `src/regime.py` — market-regime detection: rolling trend/volatility rules,
  plus a walk-forward-safe Markov-switching (Hamilton 1989) latent-state
  model via `statsmodels` (see `tests/test_regime.py`).
- `src/plots.py` — matplotlib charts for the memo (equity curves, drawdown,
  rolling Sharpe, feature importance, permutation test), written to `plots/`.
- `src/events.py` / `dashboard.py` — structured run logging and a live
  terminal dashboard (read-only; never places trades).
- `run_pipeline.py` — the end-to-end study described in the memo.
- `docs/` — textbook-style theory chapters, one per strategy.
- `tests/` — engine, risk-overlay, data-hygiene, pair-selection, feature,
  ML-pipeline, deep-learning, and regime-detection correctness checks (not
  strategy alpha claims).
