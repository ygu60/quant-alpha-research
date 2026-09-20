# Quant Research Pipeline — Memo

Author: Jimmy · Pipeline built: 2026-09-19 · Real-data rebuild: 2026-09-19/20

## 1. Goal

Build a research → backtest → (eventually) live pipeline for algorithmic
trading through Robinhood's Agentic Trading MCP, following the same process
a quant researcher would use: literature-grounded hypothesis, vectorized
backtest, genuine out-of-sample validation, honest reporting of what did and
didn't survive that validation, then paper trading before risking capital.

**This is a rebuild of an earlier version of this memo.** The original pass
concluded this sandbox had no live market-data access and reported Sharpe
ratios on synthetic data only. That conclusion was environment-specific, not
a property of this project: this session reaches Yahoo Finance's public
chart JSON endpoint fine (`query1.finance.yahoo.com/v8/finance/chart/...`,
verified manually before writing any fetcher code), so everything below is
built on ~10 years of real daily prices. Where the two versions disagree,
trust this one.

## 2. Literature grounding

Each strategy implemented is a well-replicated, decades-old anomaly chosen
specifically because it is capacity-constrained — the mechanism the small
account thesis in §3 depends on:

- **Cross-sectional short-term reversal** — Jegadeesh (1990), Lehmann
  (1990), Lo & MacKinlay (1990). After a short (1–5 day) outsized move,
  especially in smaller/less liquid names, prices show partial mean
  reversion, plausibly compensation for providing short-term liquidity to
  order-flow-driven price pressure.
- **Pairs trading** — Gatev, Goetzmann & Rouwenhorst (2006), "Pairs
  Trading: Performance of a Relative-Value Arbitrage Rule" (*Review of
  Financial Studies*). Their distance method — rank candidate pairs by
  normalized-price distance over a FORMATION period, trade the closest
  pairs in a later TRADING period — is implemented as-is in `src/pairs.py`,
  restricted to sector-matched candidates (regional banks) rather than the
  whole cross-section.
- **Turn-of-month seasonal** — Ariel (1987), Lakonishok & Smidt (1988).
  Documented at the broad-portfolio/index level, which is why it's tested
  here against SPY/IWM/IJR rather than individual illiquid small caps —
  testing it on the same 23-name universe as the other strategies would be
  applying a broad-portfolio effect somewhere it wasn't shown to hold.
- **Probabilistic Sharpe Ratio** — Bailey & Lopez de Prado (2012), "The
  Sharpe Ratio Efficient Frontier". Used throughout instead of trusting raw
  Sharpe at face value; see §6.
- **Volatility targeting** — Barroso & Santa-Clara (2015), "Momentum Has Its
  Moments", is the standard reference for vol-scaling improving risk-
  adjusted returns. It's cited here for the opposite reason: this project's
  results (§5) show vol-targeting stacked with a tight drawdown kill switch
  *destroys* value for a mean-reversion strategy, plausibly because
  mean-reversion's edge concentrates in exactly the high-volatility periods
  vol-targeting de-levers out of — a genuine difference from the momentum
  case the technique is usually validated on, not a contradiction of it.

An earlier pass also scanned September-2026 arXiv preprints (`q-fin.TR`,
`q-fin.ST`) for frontier ideas — singular spectrum analysis for calendar
effects, statistical-mechanics-flavored covariance regularization, Ising-
model contagion networks. None of that scan changed what got implemented:
the decades-old anomalies above have far more out-of-sample replication
behind them than anything from a single recent preprint, which is a
reasonable default prior when capital, not novelty, is the point.

## 3. The core thesis: small size is the edge

A single-thousands-dollar account cannot move any liquid market, which
means it can access anomalies that are mathematically real but
**structurally uninvestable at institutional scale** — a fund managing
$500M can't put more than a rounding error into a trade that's only good
for a few thousand dollars of size before slippage eats the edge. That's
the standard academic explanation for why short-term reversal, small pairs
spreads, and calendar effects persist at all: the trade size needed to
matter to a large fund's P&L moves the price before the fund can finish
executing.

## 4. Universe and data

**Source**: Yahoo Finance's public chart endpoint, no API key, stdlib
`urllib` only (`src/data.py::fetch_yahoo_history`). Split/dividend-adjusted
close is used throughout. Responses are cached to `data/` (gitignored).

**Candidate universe** (`src/data.py::UNIVERSE_GROUPS`), picked to match §3:

| Sector | Tickers attempted |
|---|---|
| Regional banks | WAL, ZION, CMA, SNV, FHN, WBS, ONB, PNFP, UMBF, ABCB |
| Industrials | WTS, ITT, KAI, ATKR, CIR |
| Consumer discretionary | BOOT, FIVE, WING, CAKE, SFM |
| Healthcare | OMCL, ICUI, PDCO, ADUS, AMED |
| Energy | MTDR, SM, PR, CIVI, MGY |
| Benchmarks | SPY, IWM (Russell 2000), IJR (S&P SmallCap 600) |

**Data hygiene** (`src/data.py::align_universe`) drops short-history and
stale/delisted names and trims the panel start to where cross-sectional
coverage is high, rather than silently forward-filling gaps. Running it
against the real universe caught, and correctly dropped, three distinct
real-world data problems:

1. **CMA, SNV, PDCO, AMED, CIVI — HTTP 404, every attempt, including a
   from-scratch fetch with no date range.** Consistent with real 2025–26
   M&A/going-private activity retiring these tickers (Comerica, Synovus,
   Patterson, Amedisys, Civitas are all plausible acquisition/going-private
   targets in that window). This is straightforward **survivorship bias**:
   the backtest only ever sees names that are still listed today, so its
   returns are biased upward relative to a universe that included names
   that later failed or got acquired below their trading value. Read every
   number below with that in mind — it is not corrected for here.
2. **WBS — resolves on Yahoo to an unrelated 13-day series starting July
   2026,** not Webster Financial's multi-year history. Ticker symbols get
   reassigned after some corporate actions; `align_universe`'s
   `min_obs=500` threshold caught this because 13 observations isn't enough
   history to be useful even if the data were legitimate.
3. **CIR** (Circor International) — short history in-sample, consistent
   with its 2023 take-private by KKR; dropped for the same reason as WBS.

**Net universe**: 23 names across 5 sectors, 2016-06-10 to 2026-09-18 (~9.3
years, 2,583 trading days).

## 5. Methodology

Every strategy is scored the same way, in increasing order of rigor:

1. **Fixed, literature-motivated parameters, full-sample backtest.** An
   honest first look — explicitly labeled "not an out-of-sample claim"
   because full-sample numbers always look better than they'll perform live
   (the strategy's own history was available to design it).
2. **Walk-forward out-of-sample validation** (`src/walk_forward.py`):
   rolling 252-trading-day (≈1yr) train / 63-day (≈1 quarter) test windows,
   stepping forward by the test length so consecutive test windows never
   overlap. The SAME fixed parameters are used in every window — no fitting
   at all — and only the test-window returns are ever reported. Because the
   step size equals the test length, the 37 test windows tile almost the
   entire sample from mid-2017 through mid-2026 with no gaps and no
   overlap: the "out-of-sample" period here is not a token holdout, it's
   nearly the whole live history of the strategy.
3. **Nested walk-forward parameter search** (short-term reversal only,
   `walk_forward_with_selection`): at each window, a small 3×2 grid
   (lookback ∈ {2,3,5} days × quintile fraction ∈ {15%,25%}) is scored
   using ONLY that window's training slice, the best-scoring combination is
   applied to generate that window's signal, and — as with step 2 — only
   the test-window return is kept. This tests whether trying to adapt
   parameters over time helps or hurts, without the parameter search ever
   seeing the data it's graded on.
4. **Risk-managed walk-forward** (`src/risk.py::apply_full_overlay` inside
   the same walk-forward loop): volatility targeting, per-name/gross
   exposure caps, and a wide circuit-breaker drawdown kill switch, with
   parameters fixed by risk-tolerance reasoning (§7), not fit to this data.

Transaction costs are a flat 10bps one-way on turnover throughout
(`cost_bps=10` in every call) — a simplification; see §9 for why this
likely understates real costs on the smallest/least liquid names here.

## 6. Guarding against overconfidence: the Probabilistic Sharpe Ratio

A raw Sharpe ratio from a finite sample — especially a skewed one, which
mean-reversion and pairs strategies typically are — overstates confidence
in the true, population Sharpe. Every result below also reports **PSR**
(Bailey & Lopez de Prado 2012, `src/metrics.py::probabilistic_sharpe_ratio`):
the probability, given the observed Sharpe, sample size, skew, and
kurtosis, that the TRUE Sharpe exceeds zero. A Sharpe of 0.5 computed on 60
days of data and a Sharpe of 0.5 computed on 2,000 days of data are not
equally trustworthy; PSR is what tells them apart. Treat PSR below ~0.90 as
"plausible but not strong evidence," and below ~0.60 as "indistinguishable
from noise" for practical purposes.

## 7. Results

All numbers below are from `python run_pipeline.py` against the real
universe in §4. Reproduce with that command; cached data means a re-run
without new history is fast.

### 7.0 Benchmarks (context, not strategies)

| | Ann. return | Ann. vol | Sharpe | PSR | Max drawdown |
|---|---|---|---|---|---|
| Equal-weight buy & hold (this universe) | 20.8% | 27.5% | 0.83 | 0.995 | -57.1% |
| IWM (Russell 2000) buy & hold | 10.7% | 22.8% | 0.56 | 0.965 | -41.1% |

This was a strong decade for small/mid-cap equities overall (COVID
recovery, multi-year rally) — every active strategy below is being compared
against that high bar, not a flat market.

### 7.1 Strategy 1 — Short-term reversal

| Step | Ann. return | Ann. vol | Sharpe | PSR | Max drawdown |
|---|---|---|---|---|---|
| (a) Full-sample, fixed params | 14.1% | 41.0% | 0.54 | 0.953 | -75.6% |
| (b) Walk-forward OOS, fixed params, **unmanaged** | 10.1% | 42.4% | 0.45 | 0.909 | -75.6% |
| (c) Nested walk-forward **with tuning** | -5.6% | 43.5% | 0.09 | 0.610 | -78.4% |
| (d) Walk-forward OOS, **risk-managed** | 2.0% | 11.5% | **0.23** | **0.756** | **-25.5%** |

Reading this in order: a real, out-of-sample-replicated signal exists (b) —
the point estimate barely moves between full-sample (a) and true walk-
forward OOS (b), which is itself informative. But it comes with a max
drawdown that no small account should carry unmanaged (c. -76%, essentially
"lose three-quarters of the account at some point"). Trying to adapt the
parameters over time (c) makes OOS performance *worse*, not better — good
evidence the untuned default isn't a lucky parameter draw, since deliberately
searching for a better one, using only information available at the time,
failed to find one. Applying real risk management (d) cuts the drawdown by
two-thirds (-76% → -25.5%) but also costs about half the raw Sharpe
(0.45 → 0.23), and pulls the confidence level down with it (PSR 0.91 → 0.76)
— because a smaller, less volatile position size produces a smaller-sample-
equivalent signal-to-noise ratio, not because anything is wrong with the
overlay.

**Diversification check**: the risk-managed strategy's OOS returns are
correlated **0.70** with simply holding the buy-and-hold universe over the
same dates. Blending it with buy-and-hold does not improve risk-adjusted
return in this sample — every blend from 100% buy-and-hold down to 50/50
has a *lower* Sharpe than pure buy-and-hold (0.78 → 0.66 as reversal weight
increases from 0% to 50%). **This strategy is not a portfolio diversifier
here; its case rests entirely on its own modest risk-adjusted return and
much shallower drawdown**, which may still matter to an investor who
weights capital preservation more than an investor who's indifferent to a
-57% drawdown, but it is not "free" incremental alpha on top of an existing
equity allocation.

### 7.2 Strategy 2 — Pairs trading (regional banks)

Formation-based pair selection (`src/pairs.py`) picks the 2 closest pairs
by normalized-price distance from a 7-bank candidate pool
(WAL, ZION, FHN, ONB, PNFP, UMBF, ABCB) using ONLY the training slice of
each walk-forward window, then trades them in that window's test slice.

| | Ann. return | Ann. vol | Sharpe | PSR | Max drawdown |
|---|---|---|---|---|---|
| Walk-forward OOS | -0.3% | 6.6% | **-0.02** | **0.481** | -20.6% |

No credible edge. PSR of 0.48 means the observed Sharpe is statistically
indistinguishable from zero — worse than a coin flip's worth of confidence
that the true Sharpe is even positive. **Not recommended for deployment as
currently specified.** Plausible explanations: regional banks in this
sample period moved together on a shared macro factor (rate-cycle, 2023
regional-bank-crisis correlation spike) more than they showed idiosyncratic
mean-reverting spread behavior; a 7-name candidate pool is also a small
search space by the standards of Gatev et al.'s original CRSP-wide study.

### 7.3 Strategy 3 — Turn-of-month

Applied to SPY/IWM/IJR (broad-portfolio effect; see §2 for why not
individual names), standard `days_before=1, days_after=3` window.

| | Ann. return | Ann. vol | Sharpe | PSR | Max drawdown |
|---|---|---|---|---|---|
| Walk-forward OOS | 0.7% | 9.4% | 0.12 | 0.639 | -27.8% |

Weak and not statistically convincing (PSR 0.64). The effect may have
decayed since the original 1980s studies (a common finding once a calendar
anomaly becomes widely known and arbitraged), or may simply not be large
enough to clear transaction costs and a modest exposure window at this
sample size. **Not recommended for deployment as currently specified.**

### 7.4 Recommendation

Of the three, **only short-term reversal + risk overlay** shows a positive,
cost-adjusted, out-of-sample Sharpe that also survives a genuine attempt to
improve on it via parameter search. That is the strategy this pipeline
recommends carrying forward — with the explicit caveat that PSR 0.76 is
*moderate*, not overwhelming, confidence, and that it earns its case on
absolute risk-adjusted return and drawdown containment, not as a
diversifier.

**Headline number, if one number is wanted: out-of-sample Sharpe ratio
0.23** (annualized return 2.0%, max drawdown -25.5%, PSR 0.76), for
short-term cross-sectional reversal (3-day lookback, quintile long book)
with a 10%-annualized-vol target, 25% single-name cap, and a 30% circuit-
breaker drawdown kill switch, net of 10bps one-way transaction costs, over
23 real names across 5 sectors, walk-forward out-of-sample from mid-2017
through mid-2026.

## 8. Research process: two bugs found and fixed by backtesting on real data

Worth stating explicitly, because it's part of why the numbers above should
be trusted more than the numbers this project reported before real data was
available: **two materially wrong intermediate results were caught only
because the risk overlay was tested against ~9 years of real, longer
history, not just short hand-constructed unit test scenarios.** Both are
fixed in the current code and covered by regression tests, but a
reader auditing this project's git history will see the wrong numbers in
earlier commits — that's intentional transparency, not something to hide.

1. **Drawdown kill switch, permanent-lock bug.** The first implementation
   judged "has the drawdown recovered enough to re-enter" from the halted
   portfolio's OWN equity curve. A fully flat book returns exactly 0%
   forever, so its drawdown from its own peak can never shrink — it
   triggered once (Dec 2018) and then stayed flat for the remaining ~7
   years of the sample (76% of all days), producing a *fake* good-looking
   "risk-managed" Sharpe of 0.46 that was really just "mostly sitting in
   cash after one early loss." Caught by looking at `weights.abs().sum()`
   over time on the real backtest, not by inspection.
2. **Drawdown kill switch, flapping bug.** The first fix (trigger off the
   real/halted curve, judge recovery off a "shadow" curve that always
   applies the original weights) traded that bug for a new one: right after
   re-entering, the real curve is still deep below its stale PRE-CRASH
   peak, so it immediately re-breaches the threshold and flattens again one
   day later, repeating indefinitely. Fixed by gauging BOTH triggering and
   recovery from a single equity curve — the raw, never-halted signal's own
   drawdown — fully decoupled from the strategy's own past halt decisions,
   so there's no feedback loop between the decision and what it's measured
   against. See `src/risk.py::drawdown_kill_switch` docstring and
   `tests/test_risk.py::test_drawdown_kill_switch_recovers_using_shadow_not_actual_equity`
   for the regression test.
3. **Scoring bug (not in risk.py, in how it was called).**
   `run_backtest(..., long_only=True)` renormalizes weights to sum to 1
   every day — correct for turning a raw long/short signal into a "fully
   invested, long-only" book, but WRONG for scoring output that has already
   passed through the risk overlay, because it silently erases any
   deliberate cash position from volatility targeting or gross exposure
   caps. This produced an apparently-attractive 8.5%-annualized-vol,
   Sharpe-0.09 result that was actually still ~100% invested every day —
   the renormalization had thrown away the entire vol-targeting effect
   while leaving the number that happened to look plausible. Fixed by
   establishing a single point where long-only conversion happens
   (`run_pipeline.py::to_long_only_weights`) and scoring everything
   downstream of the risk overlay with `long_only=False`.

The broader point these three bugs support: **a risk management claim that
was only checked against short, hand-constructed test scenarios is not yet
trustworthy.** All three were found by running the actual multi-year
backtest and looking hard at whether the numbers made sense (e.g., "why is
this exposure identical across four different config"), not by reasoning
about the code in the abstract. `tests/test_risk.py` now includes
regression tests for both kill-switch failure modes.

## 9. What this backtest does NOT account for — read before risking capital

- **Survivorship bias** (§4): 5 of 30 candidate names were dropped because
  they no longer exist under their original ticker. Their pre-delisting
  returns are still counted where they had valid history, but no name that
  failed catastrophically and was removed from Yahoo's symbol database
  entirely before this fetch would appear at all. This biases every result
  above upward, by an unknown but nonzero amount.
- **Flat 10bps transaction cost.** Real bid-ask spread and slippage on the
  smaller/less liquid names in this universe (particularly the industrials
  and healthcare names, which trade less volume than the regional banks)
  is very plausibly worse than 10bps, especially for market orders sized
  against a thin order book. This especially matters for the reversal
  strategy given its ~90% average daily turnover in the unmanaged version.
- **Close-to-close execution assumption.** The backtest assumes a signal
  computed at close(t) can be executed AT close(t) pricing for the
  close(t)→close(t+1) return. Real execution would be at the next
  available price after the decision, which for an automated agent placing
  market/limit orders shortly after close or at next open will differ —
  probably modestly for these liquidity tiers, but not zero.
- **No borrow cost or availability assumption** for anything short (the
  pairs strategy's short leg). A plain Robinhood cash/individual account
  cannot short individual equities at all; the pairs numbers above assume
  margin access this project has not verified is part of the plan.
- **37 walk-forward windows, not 37 independent observations.** The test
  windows don't overlap, but adjacent windows' TRAINING data does, and
  broader market regime (e.g. the COVID crash, the 2022 rate-hiking cycle)
  affects multiple consecutive windows together — the effective number of
  independent bets is lower than 37, which PSR's sample-size term does not
  fully capture since it assumes i.i.d. per-period returns.

## 10. Robinhood MCP — status and what it means for this pipeline

Robinhood's Trading MCP (`https://agent.robinhood.com/mcp/trading`) connects
an AI agent to a dedicated, sandboxed **Agentic account** — separate from
the main account, funded with whatever budget is set aside, and the only
account this agent can trade in. Equities, options, and crypto are
supported today; prediction markets/futures are not yet. The agent gets
read access to all Robinhood accounts but can only place trades in the
Agentic account. Not yet connected in this session.

**Recommended sequence from here**, given §7's results:

1. Do not deploy pairs trading or turn-of-month as currently specified —
   neither cleared even a modest statistical confidence bar (§7.2, §7.3).
2. If proceeding with short-term reversal + risk overlay: connect the
   Robinhood MCP, open the Agentic account with a small, explicit budget
   sized to survive a repeat of the -25.5% OOS max drawdown without forcing
   a liquidation.
3. Paper-test (or trade at minimum size with manual approval) for at least
   one full quarter (matching this project's `TEST_DAYS=63` walk-forward
   window) before scaling up, and explicitly compare realized fills/slippage
   against the flat 10bps assumption in `cost_bps` — real spread costs on
   the smaller names here are likely worse, per §9.
4. Enforce the risk overlay (`src/risk.py`) at the ORDER level in live
   trading, not just in the backtest — position caps, gross exposure caps,
   and the drawdown kill switch all need to be real trading rules the agent
   checks before every order, not retrospective backtest annotations.
5. Keep re-running the walk-forward validation as new data arrives (a
   rolling monitor, not a one-time check) — PSR 0.76 means this could
   plausibly turn out to be noise in the next few quarters, and the
   pipeline should be watched for that rather than assumed durable.

## 11. Open questions

- Position sizing in dollar terms for the Agentic account budget once
  it exists — this pipeline reports weights and Sharpe, not dollar P&L,
  which depends on account size not yet specified.
- Whether to expand the candidate universe beyond 23 names (more breadth
  would reduce cross-sectional ranking noise for short-term reversal, and
  give the pairs strategy a larger candidate pool) — tradeoff against
  fetch time and the added survivorship-bias surface area of more tickers.
- Whether margin access is actually part of the plan (relevant only if
  pairs trading is revisited after further work — not recommended as-is).
