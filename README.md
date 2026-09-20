# quant-alpha-research

Research/backtest pipeline for statistical trading strategies sized for a
small (single-thousands-dollar) account, built toward eventual live testing
via Robinhood's Agentic Trading MCP.

Start with `RESEARCH_MEMO.md` — it covers the strategy thesis, the arXiv
literature scan, the current data limitation (no live market data in this
sandbox — see memo §5), and next steps.

## Quickstart

```bash
pip install -r requirements.txt
python run_pipeline.py          # runs on synthetic data — see memo §5
python -m tests.test_pipeline    # sanity checks on the backtest engine
```

## Layout

- `src/data.py` — CSV loader (`load_csv_universe`) + synthetic data generator
  used only to validate the engine.
- `src/strategies.py` — short-term reversal, pairs z-score, turn-of-month.
- `src/backtest.py` — vectorized backtest with cost model and look-ahead
  protection (weights are shifted forward one day before being applied).
- `src/walk_forward.py` — rolling out-of-sample validation.
- `src/metrics.py` — Sharpe, Sortino, max drawdown, turnover, hit rate.
- `tests/test_pipeline.py` — engine correctness checks (not strategy alpha).
