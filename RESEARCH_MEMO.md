# Quant Research Pipeline — Memo

Author: Jimmy · Pipeline built: 2026-09-19

## 1. Goal

Build a research → backtest → (eventually) live pipeline for algorithmic
trading through Robinhood's Agentic Trading MCP, following the same steps a
quant researcher would use: literature/idea generation, hypothesis, vectorized
backtest, out-of-sample validation, then paper trading before risking capital.

This memo covers what's built, what was found in the literature scan, the
core strategic thesis (small account size as an edge), and exactly what's
missing before this could touch real money.

## 2. Literature scan (arXiv, Sept 2026)

Scanned `q-fin.TR` (Trading & Market Microstructure), `q-fin.ST` (Statistical
Finance), and `physics.soc-ph` (Physics and Society, for econophysics
crossover work) for anything suggesting exploitable, under-crowded structure.
Most relevant hits:

- **["Seasonal Trading in Commodity Futures: Evidence from Regression and
  Singular Spectrum Signals"](https://arxiv.org/abs/2609.12227)** (Kosch &
  Forsberg) — uses singular spectrum analysis (SSA) to extract seasonal
  components from futures prices and trade them. Directly actionable idea:
  SSA/spectral decomposition applied to small-cap equities to find calendar
  structure beyond the simple turn-of-month effect already implemented here.
- **["Special Markowitz: Thermodynamic Formalism for the Joint
  Regularisation of Returns and Covariance"](https://arxiv.org/abs/2609.14029)**
  (Reinhardt) — a statistical-mechanics-flavored regularization of
  mean-variance optimization. Relevant once we're combining multiple signals
  into one portfolio (see §5) — reduces overfitting to noisy covariance
  estimates, which matters more, not less, with a small number of small-cap
  names.
- **["Financial Contagion Networks as Annealing-Ready Ising
  Systems"](https://arxiv.org/abs/2609.17415)** (Tomar et al.) — Ising-model
  treatment of network contagion/bailout dynamics. Interesting for systemic
  risk framing, less directly tradeable at retail size.
- **["Herding and Liquidity in Order-Book Markets
  III"](https://arxiv.org/abs/2609.20192)** (Novotny) — endogenous liquidity
  crisis dynamics under leverage; mostly HFT/market-maker relevant, not
  retail-actionable.
- Several ML forecasting papers (WaVeFuse, VertiFuseX) propose deep-learning
  index forecasters — high complexity, low transparency, and index-level
  edges are the most heavily arbitraged of all; deprioritized for now.

None of these are "free money" — they're pointers to where current academic
attention is, which is a reasonable proxy for where the frontier of
public-domain technique sits. The actual strategies implemented below lean
more on well-replicated, decades-old anomalies than on any single new paper,
because those are the ones with the most robust supporting evidence.

## 3. The core thesis: small size is the edge

You flagged the key constraint correctly: a single-thousands-dollar account
cannot move any liquid market, which means it can access a set of anomalies
that are mathematically real but **structurally uninvestable at
institutional scale** — a fund with $500M can't put more than a rounding
error into a trade that's only good for $2-3k of size before slippage eats
it. That's not a loophole, it's the standard academic explanation for why
these anomalies persist:

- **Short-term reversal** (Lehmann 1990; Jegadeesh 1990): after a sharp 1-5
  day move, especially in small/illiquid names, prices show partial
  mean-reversion. Large funds trading this need enormous breadth (thousands
  of names) to matter to their P&L, which reintroduces the market-impact
  problem they're trying to avoid.
- **Pairs / stat-arb on small, correlated names**: two thinly-traded but
  economically linked stocks (e.g. regional bank peers) can drift apart by a
  few percent and mean-revert. The dollar profit per trade is small — exactly
  why it's not worth a big desk's operational overhead, and exactly the size
  a small account trades naturally.
- **Calendar effects** (turn-of-month, day-of-week): small, well-documented,
  costly to harvest at scale relative to the edge size.

Implemented in `src/strategies.py`:

| Strategy | File function | Long/short? |
|---|---|---|
| Cross-sectional short-term reversal | `short_term_reversal` | works long-only or long/short |
| Pairs z-score mean reversion | `zscore_pairs` | long/short (needs margin) |
| Turn-of-month seasonal | `turn_of_month` | long-only |

**Important constraint**: a plain Robinhood cash/individual account cannot
short individual equities. `run_backtest(..., long_only=True)` models that
reality by clipping negative weights and renormalizing — always look at the
long-only numbers as the realistic case unless margin is explicitly part of
the plan.

## 4. What's built

```
quant-alpha-research/
  src/
    data.py          # CSV loader + synthetic universe generator
    strategies.py     # short_term_reversal, zscore_pairs, turn_of_month
    backtest.py       # vectorized backtest, cost model, long-only clipping
    walk_forward.py   # rolling train/test out-of-sample validation
    metrics.py        # Sharpe, Sortino, max drawdown, turnover, hit rate
  tests/
    test_pipeline.py  # sanity checks: no look-ahead, cost drag, Sharpe math
  run_pipeline.py     # end-to-end demo script
```

`tests/test_pipeline.py` passes 4/4 checks confirming: signals computed at
close(t) only affect the t→t+1 return (no look-ahead), transaction costs
strictly reduce returns, the Sharpe calculation matches a manual computation,
and long-only clipping never leaks a negative weight through.

## 5. ⚠️ Data limitation — read before trusting any number

**This sandbox currently has no outbound network access to market data
providers.** Yahoo Finance, Stooq, and FRED were all tested and either
blocked outright or returned empty/binary content the tools can't parse.
`run_pipeline.py` therefore runs on `simulate_universe()` — a synthetic
8-asset universe with a deliberately injected mean-reversion component, used
**only** to prove the backtest engine's plumbing is correct (see the passing
sanity tests). The Sharpe ratios it prints (mostly negative, some walk-forward
windows positive) are meaningless as a statement about real alpha — the
synthetic mean-reversion signal I injected is small relative to noise, which
is itself a realistic reminder that these edges are subtle even when real.

**To get a real answer, one of these needs to happen:**

1. You export historical daily OHLCV CSVs (Yahoo Finance's "Download" button,
   or Stooq's export) for a candidate small-cap universe and hand them to me
   — `load_csv_universe()` is ready to ingest them.
2. You have an API key for a data vendor (Polygon.io, Alpha Vantage, Tiingo,
   IEX Cloud) — I can write a fetcher against it in a follow-up.
3. Once the Robinhood Trading MCP is connected, its read tools may expose
   enough (quotes, watchlists, positions) for *forward* paper testing, though
   likely not the years of daily history needed for a proper backtest.

## 6. Robinhood MCP — status and what it means for this pipeline

Robinhood's Trading MCP (`https://agent.robinhood.com/mcp/trading`) is real
and officially supported (launched 2026), connecting an AI agent to a
dedicated, sandboxed **Agentic account** — separate from your main account,
funded with whatever budget you set aside, and the only account your agent
can trade in.

- **Supported today**: equities, options, and crypto.
- **Not yet supported**: prediction markets. Robinhood has publicly said
  event contracts / prediction markets / futures are "coming soon" to
  agentic trading but hasn't given a date — worth revisiting, but not
  buildable today. [Source: Robinhood agentic trading coverage, TechCrunch/Finder, Sept 2026]
- Your agent gets **read access to all your Robinhood accounts** (balances,
  positions, order history) but can only **place trades in the Agentic
  account** specifically.
- You haven't connected it yet. When ready: Claude Desktop → Settings →
  Connectors → Add custom connector → `https://agent.robinhood.com/mcp/trading`.

**Recommended sequence from here** (matches what you asked for — research,
backtest, then automated live testing):

1. Get real historical data in (§5) and re-run the backtest/walk-forward on
   an actual small-cap universe. Only trust a strategy that holds up
   out-of-sample across multiple non-overlapping walk-forward windows.
2. If a strategy clears that bar, connect the Robinhood MCP and open the
   Agentic account with a small, explicit budget.
3. Paper-test first even though the Agentic account is real money: run the
   strategy with either manual trade approval or a hard position-size cap for
   a few weeks and compare live fills/slippage against the backtest's cost
   assumptions (`cost_bps` in `backtest.py`) — real spread costs on small caps
   are often worse than a flat 10bps assumption.
4. Only then consider fully automatic execution, and even so, keep a
   portfolio-level max-drawdown kill switch.

## 7. GitHub

I don't have a GitHub connector available in this session and this sandbox
has no outbound network access to github.com (confirmed — `git ls-remote`
to GitHub failed at the proxy level), so I can't push this for you directly.
The project folder is ready to go — from a terminal on your own machine:

```bash
cd quant-alpha-research
git init
git add .
git commit -m "Initial quant research pipeline: backtest engine + 3 candidate strategies"
gh repo create <your-repo-name> --private --source=. --push
# (or create the repo on github.com first, then: git remote add origin <url> && git push -u origin main)
```

## 8. Open questions for you

- Which small-cap universe do you actually want to test (sector, market-cap
  band, exchange)? That determines what CSVs to pull.
- Do you want to explore margin (long/short) at all, or keep this strictly
  cash-account long-only given the account size?
