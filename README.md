# quant-alpha-research

Research/backtest pipeline for statistical trading strategies sized for a
small (single-thousands-dollar) account, built toward eventual live testing
via Robinhood's Agentic Trading MCP.

Start with `RESEARCH_MEMO.md` — it covers the literature grounding, the real
data universe and its data-quality quirks, walk-forward methodology, full
results per strategy (including two the pipeline does NOT recommend
trading), two risk-overlay bugs found and fixed during development, and
honest limitations to read before risking capital.

**Headline result** (see memo §7.4 for full context): short-term
cross-sectional reversal with a volatility-targeting risk overlay, walk-
forward out-of-sample Sharpe ratio **0.23** (PSR 0.76, max drawdown -25.5%)
over 23 real names, 2016–2026, net of transaction costs. Pairs trading and
turn-of-month were also tested and are NOT recommended as currently
specified — no credible out-of-sample edge.

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
- `src/events.py` / `dashboard.py` — structured run logging and a live
  terminal dashboard (read-only; never places trades).
- `run_pipeline.py` — the end-to-end study described in the memo.
- `tests/` — engine and risk-overlay correctness checks (not strategy
  alpha claims).
