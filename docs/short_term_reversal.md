# Cross-Sectional Short-Term Reversal

## 1. Motivation

Hold any diversified basket of stocks and rank them, each week, by how they
did the week before. A robust empirical regularity — documented across
decades and markets — is that last period's biggest losers tend to
outperform last period's biggest winners over the *next* short horizon (a
few days to a few weeks), even though the opposite pattern (momentum) shows
up at 3-12 month horizons. This chapter is about the short end of that
spectrum: 1- to 5-day reversal.

Three economic mechanisms are usually offered, and they are not mutually
exclusive:

1. **Bid-ask bounce.** If a stock trades alternately at the bid and the ask,
   consecutive *transaction-price* returns are mechanically negatively
   autocorrelated even if the *true* (unobserved) value never moves. Roll
   (1984) showed how to back out an implied effective spread from exactly
   this induced negative autocovariance. This is a pure microstructure
   artifact, not "mispricing" in any economic sense, and it is largest in
   less liquid names with wider spreads — which is one reason this project
   restricts to a smaller-cap universe rather than mega-caps, where the
   spread is small enough that bounce-driven reversal is negligible.
2. **Inventory risk and liquidity provision.** A market maker or other
   liquidity supplier who absorbs a large sell order needs to be
   compensated for the inventory risk of holding the position until it can
   be unwound; that compensation shows up as a temporary price concession
   that reverts once the order-flow imbalance clears. This is a genuine
   risk premium for providing a service (immediacy), not overreaction.
3. **Behavioral overreaction.** Investors overreact to salient short-term
   news or order flow, pushing price temporarily away from fundamental
   value; the correction back is the reversal. This is the reading Lehmann
   (1990) and Jegadeesh (1990) lean toward, without being able to fully
   separate it from (1) and (2) empirically.

All three mechanisms share a property that matters enormously for *who can
trade on them*: they are transient, small in dollar terms per trade, and
they get arbitraged away exactly in proportion to how much capital chases
them. A market maker or a large fund executing this strategy at
institutional size *becomes* the liquidity event that erodes the very
temporary mispricing it's trying to harvest. That is the capacity-
constraint argument this project's thesis (RESEARCH_MEMO.md §3) rests on.

## 2. A model that produces the effect

Suppose asset $i$'s return at time $t$ decomposes into a common (market)
factor and an idiosyncratic part with a mean-reverting overlay:

$$
R_{i,t} = \mu_i + \beta_i f_t + \varepsilon_{i,t}, \qquad
\varepsilon_{i,t} = u_{i,t} - \phi\, u_{i,t-1}, \quad \phi \in (0, 1)
$$

where $f_t$ is a common shock, $u_{i,t}$ is an i.i.d. idiosyncratic
innovation, and the $-\phi\, u_{i,t-1}$ term is the reversal overlay: part
of yesterday's idiosyncratic shock mechanically cancels today (this is the
same MA(1)-style construction used to inject a reversal signal into the
synthetic validation data in `src/data.py::simulate_universe`, for exactly
this reason — it's the simplest process with the property we want to test
for). The own-return autocovariance at lag 1 is

$$
\text{Cov}(R_{i,t}, R_{i,t-1}) = \text{Cov}(\varepsilon_{i,t}, \varepsilon_{i,t-1}) = -\phi\,\sigma_u^2 < 0.
$$

Negative own-autocovariance is exactly what a reversal strategy is trying
to monetize: if $\varepsilon_{i,t-1}$ was unusually negative, $R_{i,t}$'s
conditional mean tilts positive.

## 3. From the model to the trading rule: a contrarian portfolio

Define the equal-weighted cross-sectional average return
$\bar R_{t} = \frac{1}{N}\sum_i R_{i,t}$, and consider a contrarian
portfolio that weights each asset in proportion to how far *below* the
cross-sectional average it was last period:

$$
w_{i,t} = -\frac{1}{N}\left(R_{i,t-1} - \bar R_{t-1}\right)
$$

(negative sign: underperformers get positive weight). `short_term_reversal`
in `src/strategies.py` is a discretized, long-only version of exactly this
rule — instead of a continuous weight proportional to the return
deviation, it rank-sorts on the trailing `lookback`-day return
($k=3$ days in this project's default) and takes an equal-weighted
long position in the bottom quintile (`n_long = round(0.20 \times N)`),
which is a robust (outlier-resistant) discretization of the same idea.

**Lo & MacKinlay's decomposition (1990, "When Are Contrarian Profits Due
to Stock Market Overreaction?", *Review of Financial Studies*)** asks a
sharper question than "does this make money": *why*, algebraically, does
it make money? Write out the expected profit of the continuous-weight
version, $E[\pi_t] = \sum_i E[w_{i,t}\,R_{i,t}]$. Expanding
$w_{i,t}$ and collecting terms (the full derivation is in their paper;
Campbell, Lo & MacKinlay's *The Econometrics of Financial Markets*, §2.2,
reproduces it) decomposes the expected profit into three pieces, up to an
overall normalizing constant that depends on $N$:

$$
E[\pi_t] \;\propto\; -\underbrace{\frac{1}{N}\sum_i \text{Cov}(R_{i,t}, R_{i,t-1})}_{\text{(A) own-return autocovariance}} \;+\; \underbrace{\frac{1}{N^2}\sum_i\sum_{j\neq i} \text{Cov}(R_{i,t}, R_{j,t-1})}_{\text{(B) cross-serial covariance}} \;+\; \underbrace{\frac{1}{N}\sum_i (\mu_i - \bar\mu)^2}_{\text{(C) cross-sectional variance in means}}
$$

Term (A) is what §2's model produces: negative own-autocovariance
contributes *positively* to contrarian profit (the minus sign in front
flips the sign of a negative covariance). This is the "genuine reversal"
story. But term (B) is the paper's central, somewhat unsettling
contribution: **a positive average cross-serial covariance — stock $j$'s
return today predicting stock $i$'s return tomorrow, a lead-lag effect
with no reversal in any individual stock's own returns — contributes
*exactly as much* to measured contrarian profit as true own-stock
reversal does.** A contrarian strategy can be profitable in a market where
every individual stock is a random walk, purely because stocks lag each
other. Term (C) is a static cross-sectional dispersion effect unrelated to
timing at all (it rewards long-run winners/losers being persistently
different, not short-term reversal). Lo & MacKinlay's empirical finding
for U.S. equities was that (B), not (A), does most of the work — meaning a
lot of measured "reversal profit" in a broad-market strategy is really
a lead-lag/cross-autocorrelation phenomenon (itself plausibly a
market-microstructure artifact: large, frequently-traded stocks
impound information faster than small, thinly-traded ones, so a small
stock's return today can be partly *predictable* from a large stock's
return yesterday without either stock's own returns being autocorrelated
at all).

This matters directly for how to read this project's result: the strategy
here restricts to a moderately-sized universe (23 names, not the whole
market) with a *short* lookback (3 days) and *no* attempt to separate (A)
from (B) empirically — an honest reading of the backtest result (§5 below)
is "a contrarian rule on this universe made money out-of-sample," not "we
have identified genuine mean-reversion in individual stock returns." Which
mechanism is actually responsible is a strictly harder empirical question
this project does not answer.

## 4. What the hypothesis predicts, and how it was tested

**Falsifiable prediction:** a portfolio that is long the bottom quintile of
trailing 3-day cross-sectional returns, rebalanced daily, should show a
positive average return net of transaction costs, out of sample, on a
universe of moderately-liquid (not mega-cap, not micro-cap/illiquid-to-the-
point-of-uninvestable) equities.

**Implementation** (`src/strategies.py::short_term_reversal`): ranks assets
by trailing `lookback`-day return each day (`lookback=3` — matching the
1-5 day horizon in the Lehmann/Jegadeesh literature), goes long the
bottom `n_long` names and (in the long/short variant) short the top
`n_short`, both sized as an equal-weighted quintile of the 23-name universe
(`n_long = n_short = round(0.20 \times 23) = 5`). The long/short version
is shown for reference only; the tradeable version clips negative weights
to zero and renormalizes the long leg to full investment
(`run_pipeline.py::to_long_only_weights`), since a plain cash brokerage
account cannot short individual equities.

**Validation** (`src/walk_forward.py::walk_forward` and
`walk_forward_with_selection`): the fixed rule is walked forward across 37
non-overlapping quarterly out-of-sample windows (252-day train / 63-day
test) with no parameter fitting, and separately with a small train-only
parameter search (lookback $\in \{2,3,5\}$ days $\times$ quintile fraction
$\in \{15\%, 25\%\}$) to check whether adapting the parameters over time
helps. A **risk overlay** (`src/risk.py::apply_full_overlay`) — volatility
targeting to a 10% annualized budget, a 25% single-name cap, and a wide
(30%) drawdown circuit breaker — is then layered on top with parameters
fixed by risk-tolerance reasoning, not fit to the backtest.

## 5. Empirical postscript — what this project actually found

Out-of-sample, walk-forward, on the real 23-name universe (2016-06 to
2026-09): the untuned rule shows Sharpe **0.45** (PSR 0.91) but an
unmanaged max drawdown of **-75.6%** — far too much risk for a small
account to carry directly. The train-only parameter search made
out-of-sample performance *worse* (Sharpe 0.09), evidence the untuned
default is not a fragile, lucky parameter draw. With the risk overlay
applied, Sharpe drops to **0.23** (PSR 0.76) but max drawdown improves to
**-25.5%** — a real, if modest, edge that trades raw Sharpe for
survivability. Its out-of-sample returns are correlated 0.70 with simply
holding the universe, so — despite the theoretical story above being about
a genuinely different (short-horizon, contrarian) mechanism from buy-and-
hold — empirically it behaves as a lower-volatility variant of long
exposure to the same names more than as an independent source of return.
See RESEARCH_MEMO.md §7.1 for the full numbers and caveats (survivorship
bias, transaction-cost assumptions, sample-size limitations on the
walk-forward windows).

## References

- Lehmann, B. N. (1990). "Fads, Martingales, and Market Efficiency."
  *Quarterly Journal of Economics*, 105(1), 1-28.
- Jegadeesh, N. (1990). "Evidence of Predictable Behavior of Security
  Returns." *Journal of Finance*, 45(3), 881-898.
- Lo, A. W., & MacKinlay, A. C. (1990). "When Are Contrarian Profits Due to
  Stock Market Overreaction?" *Review of Financial Studies*, 3(2), 175-205.
- Jegadeesh, N., & Titman, S. (1993). "Returns to Buying Winners and
  Selling Losers: Implications for Stock Market Efficiency." *Journal of
  Finance*, 48(1), 65-91. (The momentum counterpart at 3-12 month horizons
  — useful contrast: the sign of the cross-sectional autocorrelation
  flips between the reversal and momentum horizons.)
- Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask
  Spread in an Efficient Market." *Journal of Finance*, 39(4), 1127-1139.
- Campbell, J. Y., Lo, A. W., & MacKinlay, A. C. (1997). *The
  Econometrics of Financial Markets*. Princeton University Press — Ch. 2
  reproduces the Lo-MacKinlay contrarian-profit decomposition in full.
