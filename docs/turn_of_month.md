# The Turn-of-Month Effect

## 1. Motivation

Unlike short-term reversal or pairs trading, the turn-of-month effect
makes no claim about mispricing at the individual-stock level at all. It
is a **calendar anomaly**: the observation that a disproportionate share
of the broad stock market's average return arrives in a narrow window
around the transition from one calendar month to the next — historically,
roughly the last trading day of the month through the first few trading
days of the following month — with the remaining days of the month
showing, on average, close to zero excess return.

The proposed mechanisms are institutional and flow-driven rather than
behavioral-individual-stock-level:

- **Payroll and pension-contribution flows.** Many retirement and
  savings-plan contributions (401(k) deferrals, pension fund inflows) are
  processed on a monthly cycle and hit the market at or near month-end,
  creating a recurring wave of price-insensitive buying demand.
- **Institutional rebalancing and window dressing.** Fund managers
  rebalancing to month-end targets, or adjusting reported holdings ahead
  of month-end statements, can generate systematic month-end order flow
  independent of any new fundamental information.
- **Settlement and liquidity-cycle effects.** Some documented calendar
  patterns in the literature (day-of-week, holiday effects) share a family
  resemblance with turn-of-month in being attributed to institutional
  plumbing rather than to any single firm's fundamentals — the common
  thread is that flows tied to a fixed calendar date needn't have anything
  to do with information.

Because the proposed mechanism operates at the level of *aggregate market
demand*, not any individual company's story, the effect is tested and
motivated at the **broad-portfolio or index level** — which is why this
project applies it to SPY/IWM/IJR rather than to the 23-name individual-
stock universe used for the other two strategies. Testing a market-wide
liquidity-flow effect on a handful of individual small/mid-cap names would
be applying the theory somewhere it was never claimed to hold.

## 2. Formal setup: an event-study regression

The standard way the empirical calendar-anomaly literature tests a claim
like this is a simple **dummy-variable (event-study) regression**. Let
$R_t$ be the market's return on day $t$, and define an indicator

$$
D_t = \begin{cases} 1 & \text{if day } t \text{ falls in the turn-of-month window} \\ 0 & \text{otherwise} \end{cases}
$$

and regress

$$
R_t = \alpha + \beta\, D_t + \epsilon_t.
$$

Under the null hypothesis of no turn-of-month effect, $\beta = 0$: the
turn-of-month days have the same expected return as any other day. The
economically interesting quantities are $\hat\alpha$ (the average return
on a "normal" day) and $\hat\alpha + \hat\beta$ (the average return on a
turn-of-month day); the classic, striking finding in this literature (see
references) is that essentially all of the market's average return is
concentrated in the $D_t=1$ days — i.e. $\hat\beta$ is large and
statistically significant, and the average return on the *complement* of
the turn-of-month window is close to zero.

Significance is assessed the ordinary way: $t = \hat\beta / \text{SE}(\hat\beta)$,
compared against a standard normal or $t$-distribution critical value
(with the usual caveat that daily returns are not i.i.d. — heteroskedasticity
and autocorrelation in $\epsilon_t$ mean a naive OLS standard error is not
strictly valid without a Newey-West or similar correction, though the
literature's core finding has proven robust to that adjustment). This
project's own validation approach (§4) is a variant of the same idea
implemented as a trading rule and evaluated by realized Sharpe ratio
rather than a regression $t$-statistic, but the underlying logical
structure — is the mean return inside the window different from the mean
return outside it, more than sampling noise would explain — is the same.

## 3. From the regression to the trading rule

The dummy-variable framework translates directly into a trading rule:
hold a fully-invested, diversified long position **only** while $D_t = 1$,
and hold cash (or nothing) otherwise. `src/strategies.py::turn_of_month`
implements $D_t$ with `days_before=1` (the last trading day of the current
month) through `days_after=3` (the first three trading days of the next
month) — a 4-trading-day window per month-end, matching the standard
window used in the literature this project cites. Because the position is
either fully on or fully off, and only ever long (never short), the
"strategy" is really just a **conditional exposure timer** on top of
whatever underlying instrument is held — here, an equal-weighted blend of
SPY, IWM, and IJR.

## 4. What the hypothesis predicts, and how it was tested

**Falsifiable prediction:** a portfolio that is invested in a broad index
ETF only during the 4-trading-day turn-of-month window (and in cash
otherwise) should show a positive Sharpe ratio out of sample, net of
transaction costs, capturing most of the index's average return while
being invested only a small fraction of total trading days.

**Implementation**: `turn_of_month(bench_prices, days_before=1,
days_after=3)` applied to the SPY/IWM/IJR benchmark panel, walked forward
through the same 252-day train / 63-day test scaffolding as every other
strategy in this project (`src/walk_forward.py::walk_forward`) — though
because the rule has no fitted parameters at all (the calendar window is
fixed by convention, not estimated from data), the "walk-forward" step here
is really just confirming the fixed rule's out-of-sample stability through
time rather than guarding against overfitting a parameter.

## 5. Empirical postscript — what this project actually found

Out-of-sample, walk-forward: Sharpe **0.12**, PSR **0.64**, max drawdown
-27.8%. This is weak and not statistically convincing by this project's
own stated bar (PSR below ~0.90 is "plausible but not strong evidence").
**Not recommended for deployment as currently specified.** Two
non-exclusive readings: the effect may simply have decayed — a common
finding once a calendar anomaly is well-documented and enough capital
starts trading around it specifically to compress the mispricing it
implies (the same capacity-and-crowding logic that appears throughout this
project, here working in the *other* direction: turn-of-month, unlike
short-term reversal in a small-cap universe, is trivially tradeable at
enormous scale by anyone holding index futures, so it should be *more*
arbitraged away over time, not less); or the effect may simply be too
small in magnitude, on this specific decade of SPY/IWM/IJR data, to clear
a 10bps transaction-cost drag applied twice a month (entry and exit) over
a modest number of independent event windows. See RESEARCH_MEMO.md §7.3
for the full numbers.

## References

- Ariel, R. A. (1987). "A Monthly Effect in Stock Returns." *Journal of
  Financial Economics*, 18(1), 161-174.
- Lakonishok, J., & Smidt, S. (1988). "Are Seasonal Anomalies Real? A
  Ninety-Year Perspective." *Review of Financial Studies*, 1(4), 403-425.
- Newey, W. K., & West, K. D. (1987). "A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent
  Covariance Matrix." *Econometrica*, 55(3), 703-708. (The standard-error
  correction referenced in §2 for testing a mean-difference hypothesis on
  non-i.i.d. daily returns.)
