# Pairs Trading: A Relative-Value Statistical Arbitrage

## 1. Motivation

Pairs trading asks a narrower, more defensible question than "will this
stock go up": *will the price relationship between these two stocks
revert toward its historical norm?* Two firms with similar business models
and shared risk exposures — two regional banks funded by the same deposit
base and exposed to the same regional credit cycle, say — should have
returns driven substantially by a common factor. If a firm-specific shock
(an earnings surprise, a rumor, a large uninformed sell order) pushes one
of them away from the other temporarily, a bet that the *spread* reverts
is a bet on a much narrower, more identifiable phenomenon than a directional
bet on either stock, and it hedges away the shared market/sector factor
almost entirely.

This is a **relative-value** strategy: it does not require forecasting
whether banks go up or down, only that two economically-linked names that
have drifted apart will converge again. The economic mechanisms usually
cited:

- **Temporary liquidity/order-flow shocks** that hit one name and not its
  peer, absent any real change in relative fundamentals.
- **Slow information diffusion** — news that is fundamentally relevant to
  both names gets priced into one faster than the other (the more heavily
  covered/traded of the pair usually leads).
- **Limits to arbitrage at scale.** Precisely because the profit per pair
  is a few percent on a modest position, and a fund needs to run *many*
  such pairs to matter to its overall P&L, large capital chasing this
  strategy needs deep, liquid pair inventories — which pushes big funds
  toward large-cap pairs, leaving smaller-cap, thinner pairs relatively
  under-arbitraged. This is the same capacity-constraint logic as
  short-term reversal (see `docs/short_term_reversal.md` §1), applied to a
  different anomaly.

## 2. Formal setup: the spread as a mean-reverting process

Let $P_{A,t}$ and $P_{B,t}$ be the prices of two economically linked
assets, and define the **log-spread**

$$
S_t = \log P_{A,t} - \log P_{B,t}.
$$

The working assumption is that $S_t$ is (at least approximately, over the
horizons traded) a **stationary, mean-reverting process** — not that
either $\log P_{A,t}$ or $\log P_{B,t}$ individually is stationary (they
are not; they're integrated price processes). The canonical continuous-time
model for a mean-reverting spread is the **Ornstein-Uhlenbeck (OU)
process**:

$$
dS_t = \theta(\mu - S_t)\,dt + \sigma\, dW_t
$$

where $\theta > 0$ is the speed of mean reversion, $\mu$ is the long-run
equilibrium spread, and $W_t$ is a standard Brownian motion. This process
has a known stationary distribution, $S_\infty \sim \mathcal N(\mu,
\sigma^2/2\theta)$, which is exactly what licenses a **z-score** trading
rule: standardize the spread by its own (estimated) stationary mean and
standard deviation,

$$
z_t = \frac{S_t - \hat\mu}{\hat\sigma}
$$

and treat $|z_t|$ large as "the spread is unusually far from its
equilibrium, bet on reversion." `src/strategies.py::zscore_pairs` computes
exactly this $z_t$ using a rolling estimation window (`lookback=20` days
for $\hat\mu, \hat\sigma$), enters a position when $|z_t|$ exceeds
`entry_z` (long the spread — long $A$, short $B$ — when $z_t < -\text{entry\_z}$,
i.e. $A$ is cheap relative to $B$), and exits when $|z_t|$ falls back
below `exit_z`.

**Why an entry/exit band rather than a single threshold?** An OU
process's *first-passage time* back to its mean, starting from a given
distance, is itself a random variable with substantial variance — trading
on every single zero-crossing (`exit_z = 0`) generates enormous turnover
(and transaction-cost drag) for a process that oscillates around, rather
than converges monotonically to, its mean. The band ($\text{entry\_z}=2.0$,
$\text{exit\_z}=0.5$ in this project) is a standard practical fix: exit
"close enough" to the mean rather than waiting for an exact crossing.

## 3. Which pair? The Gatev-Goetzmann-Rouwenhorst distance method

An OU process is a *model*; nothing forces any given pair of stocks to
actually follow one. The pair-selection problem is: out of a candidate
universe, which pairs are plausible enough to trade? A tempting-but-wrong
approach is to look at the whole history, see which pair traded profitably,
and declare that the "chosen" pair — this is a textbook overfitting
mistake, selecting on the outcome you are about to report.

**Gatev, Goetzmann & Rouwenhorst (2006, "Pairs Trading: Performance of a
Relative-Value Arbitrage Rule," *Review of Financial Studies*)** fix this
by splitting time into a **formation period** and a later, disjoint
**trading period**. Pairs are selected using *only* formation-period data,
by minimizing the sum of squared deviations between the two assets'
*normalized* price paths:

$$
D(A, B) = \sum_{t \in \text{formation}} \left(\tilde P_{A,t} - \tilde P_{B,t}\right)^2, \qquad \tilde P_{i,t} = \frac{P_{i,t}}{P_{i,t_0}}
$$

(each price series rebased to 1.0 at the start of the formation window, so
the distance is scale-free and comparable across candidate pairs). The
$\text{top-}n$ pairs by *smallest* $D$ are then traded, unchanged, over the
following trading period.

**This is a proxy, not the same thing as identifying the best OU process.**
Minimizing $D(A,B)$ finds the pair whose price paths tracked each other
most closely *on average* over the formation window — it does not directly
estimate or maximize the mean-reversion speed $\theta$, nor does it
guarantee the spread's variance $\sigma^2$ is small (a pair can have small
average distance but a wide, slow-reverting spread, which trades poorly
under a z-score rule). It also does not test for cointegration in the
econometric sense (e.g. an Engle-Granger or Johansen test would ask a
related but distinct question — whether a *linear combination* of the two
log-prices is stationary, allowing for a hedge ratio other than 1:1, which
the distance method implicitly assumes by working with a 1:1 log-spread).
The distance method's appeal is exactly that it is simple, requires no
distributional assumptions, and — the property this project relies on
most directly — it is trivial to keep out-of-sample: `select_pairs_by_
distance` (`src/pairs.py`) is a pure function of the formation slice, with
no look-ahead possible by construction.

`src/pairs.py::adaptive_pairs_weights` wires this into the walk-forward
loop: for each 252-day training slice of a rolling window, it restricts
candidates to a single sector (`regional_banks` — 7 names in the surviving
universe) rather than the whole cross-section, on the reasoning from §1
that two *economically linked* names having a low price distance is far
more likely to reflect a real common factor than two arbitrary names
coincidentally tracking each other. Restricting the candidate pool from
$\binom{23}{2}=253$ pairs to $\binom{7}{2}=21$ also directly shrinks the
multiple-comparisons problem: fewer candidates tested for "closeness" means
less chance that the winning pair is a spurious artifact of the formation
window's particular sample path.

## 4. What the hypothesis predicts, and how it was tested

**Falsifiable prediction:** the top-2 closest-tracking pairs (by formation-
period distance) among a pool of economically-linked regional banks should
show positive risk-adjusted returns, out of sample, when traded with a
z-score entry/exit rule.

**Implementation**: `adaptive_pairs_weights(prices, train_days=252,
candidates=<7 regional banks>, top_n=2, lookback=20, entry_z=2.0,
exit_z=0.5)`, dropped into the same `walk_forward` scaffolding as every
other strategy in this project (252-day train / 63-day test, 37
non-overlapping test windows). Pair identity is re-selected every window
using only that window's training slice — so which two banks are being
traded can, and does, change over the 9-year sample as different pairs'
formation-period distances shift.

## 5. Empirical postscript — what this project actually found

Out-of-sample, walk-forward: Sharpe **-0.02**, PSR **0.48**, max drawdown
-20.6%. A PSR of 0.48 means the observed result is statistically
indistinguishable from zero — worse, in confidence terms, than a coin
flip on whether the true Sharpe is even positive. **Not recommended for
deployment as currently specified.** Two plausible, non-exclusive
explanations: regional banks in this specific 2016-2026 sample period may
have moved together on a shared macro factor (the rate-hiking cycle, the
2023 regional-bank-stress episode, which likely *raised* their pairwise
correlation across the board rather than leaving idiosyncratic,
mean-reverting spreads to trade) more than they displayed genuine
idiosyncratic divergence-and-reversion; and a 7-name candidate pool is a
much smaller search space than Gatev et al.'s original CRSP-wide universe,
so there may simply not be enough genuinely cointegrated pairs available
within this project's sector-restricted, capacity-appropriate universe.
See RESEARCH_MEMO.md §7.2 for the full numbers.

## References

- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs
  Trading: Performance of a Relative-Value Arbitrage Rule." *Review of
  Financial Studies*, 19(3), 797-827.
- Uhlenbeck, G. E., & Ornstein, L. S. (1930). "On the Theory of the
  Brownian Motion." *Physical Review*, 36(5), 823-841. (The original OU
  process; the mean-reverting spread model in §2 is the standard
  application of this classical stochastic process to a price spread.)
- Engle, R. F., & Granger, C. W. J. (1987). "Co-Integration and Error
  Correction: Representation, Estimation, and Testing." *Econometrica*,
  55(2), 251-276. (The econometric cointegration-testing alternative to
  the distance method, mentioned in §3 for contrast — not implemented in
  this project.)
