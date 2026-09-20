# Market-Regime Conditioning

## 1. Motivation

Every strategy studied elsewhere in this project (`docs/short_term_reversal.md`,
`docs/pairs_trading.md`, `docs/turn_of_month.md`) is tested with an
**unconditional** hypothesis: "this rule has positive expected return,
averaged over all of history." That average can hide a lot. A rule with a
strong edge in some environments and a strong *negative* edge in others can
average out to a modest positive number that looks like a weak-but-real
effect, when the more accurate description is "works great sometimes,
loses money other times, and you can't currently tell which is which
without more information." Regime conditioning asks a sharper question:
does the environment itself — trending vs. range-bound, calm vs.
turbulent — predict *when* a given rule works, well enough to act on?

This is not a new idea. Two literatures motivate it directly:

1. **Regime-switching models of returns themselves.** Hamilton (1989, "A
   New Approach to the Economic Analysis of Nonstationary Time Series and
   the Business Cycle", *Econometrica*) introduced the idea that an
   economic time series can be usefully modeled as switching between a
   small number of unobserved ("latent") states — e.g. "expansion" and
   "recession" — each with its own mean and variance, with a Markov chain
   governing the (unobserved) transitions between them. Ang & Bekaert
   (2002, "International Asset Allocation with Regime Shifts", *Review of
   Financial Studies*) extended this to asset allocation, showing that
   optimal portfolio weights differ substantially across a high-volatility/
   high-correlation regime and a calm regime.
2. **The specific mechanism behind short-term reversal.** Recall from
   `docs/short_term_reversal.md` that one of the three candidate
   explanations for reversal is compensation for liquidity provision:
   absorbing an order-flow imbalance carries inventory risk, and the
   compensation for bearing that risk should be *larger*, not constant,
   when volatility (and therefore inventory risk) is elevated. If that
   mechanism is doing real work, the strategy's edge should be
   measurably concentrated in higher-volatility periods rather than spread
   evenly across all conditions — a testable, falsifiable prediction, not
   just a plausible story.

## 2. Two ways to define "regime", formally

### 2.1 Rule-based regimes

The simplest possible regime indicator is a deterministic function of
trailing price history. A trend regime,

$$
\text{Trend}_t = \mathbb{1}\left[P_t > \text{MA}_{200}(t)\right],
$$

(price above its trailing 200-day moving average) requires no estimation
at all — it's a rolling statistic, exactly as causal as any feature in
`src/features.py`. A volatility regime does the same for realized
volatility, but relative to its OWN history rather than an absolute
threshold (so it adapts across assets and eras):

$$
\text{VolPercentile}_t = \frac{1}{W}\sum_{s=t-W}^{t-1} \mathbb{1}\left[\hat\sigma_s < \hat\sigma_t\right], \qquad \hat\sigma_t = \text{std}\left(\{r_{t-k+1}, \ldots, r_t\}\right)
$$

where $\hat\sigma_t$ is realized volatility over a trailing $k$-day window
and $\text{VolPercentile}_t$ ranks today's realized vol against the
trailing $W$-day history of that same statistic. Bucketing this into
tertiles (low/mid/high) gives a simple, robust, three-state regime label
with no distributional assumptions.

### 2.2 A latent-state model: Markov-switching regression

The rule-based approach is transparent but arbitrary — why 200 days, why
tertiles? A Markov-switching model instead *estimates* the regime
structure from the data itself. Suppose returns $r_t$ are drawn from one
of $k$ regimes, indexed by an unobserved state $S_t \in \{1, \ldots, k\}$:

$$
r_t \mid S_t = j \;\sim\; \mathcal{N}(\mu_j, \sigma_j^2), \qquad
P(S_t = j \mid S_{t-1} = i) = p_{ij}
$$

The state sequence follows a first-order Markov chain with transition
matrix $P = [p_{ij}]$. Given a return series, the parameters
$\{\mu_j, \sigma_j^2, p_{ij}\}$ are estimated by maximum likelihood (the
EM algorithm, alternating between inferring state probabilities given
parameters and re-estimating parameters given those probabilities, is the
standard approach — see Hamilton's original paper). Two related but
distinct quantities come out of a fitted model:

- **Filtered probabilities**, $P(S_t = j \mid r_1, \ldots, r_t)$ — the
  probability of being in state $j$ at time $t$ using only information
  through time $t$. This is the CAUSAL quantity: it never uses $r_{t+1}$
  or later.
- **Smoothed probabilities**, $P(S_t = j \mid r_1, \ldots, r_T)$ — the
  probability of being in state $j$ at time $t$ using the FULL sample,
  including the future relative to $t$. This is more accurate as a
  historical/descriptive characterization, and unusable as a live trading
  signal (it requires knowing the future).

## 3. The walk-forward-safety problem this creates

Every other model in this project (the ML feature set, the risk overlay,
the pairs-formation step) has an obvious causal boundary: a rolling window
ending at $t$. A Markov-switching model is different because its
PARAMETERS, not just its output, are fit by maximum likelihood over
whatever sample it's given. Fitting $\{\mu_j, \sigma_j^2, p_{ij}\}$ on a
window that includes days $t+1, \ldots, t+h$ means the very definition of
"what a high-volatility regime looks like" has been informed by data the
strategy shouldn't have at time $t$ — even if you only ever *read off* the
filtered probability at time $t$ afterward.

The fix mirrors the pair-formation step in `docs/pairs_trading.md`:
separate *estimation* from *inference*.

1. **Estimate** $\hat\theta = \{\hat\mu_j, \hat\sigma_j^2, \hat p_{ij}\}$
   using only the training slice of a walk-forward window
   (`src/regime.py::fit_markov_params`).
2. **Filter** the full window (train + test) with those parameters held
   FIXED — no re-estimation — and read off the filtered (causal)
   probabilities for the test period only (`filter_high_vol_probability`).

A subtlety worth stating plainly because it was caught empirically, not
by inspection: `statsmodels`' `.fit(start_params=..., maxiter=0)` — which
looks like it should just "start from these parameters and do nothing" —
still measurably moved several parameters (a few percent on the variance
terms) when tested against a window whose test portion had been
artificially shocked. Zero iterations of the optimizer is not the same as
zero re-estimation; some implementations still take at least one internal
step. The model's `.filter(params)` method, which runs the Hamilton filter
recursion directly against a FIXED parameter vector with no optimizer
involved at all, is what actually guarantees zero leakage — verified the
same way, by shocking only the test-period data and confirming the
resulting parameters are byte-identical.

## 4. What the hypothesis predicts, and how it was tested

**Falsifiable prediction:** the short-term reversal strategy's excess
return, conditional on a high-volatility environment, should exceed its
unconditional excess return — and conversely, restricting to calm periods
should underperform (or a symmetric prediction: performance should not be
uniform across regimes at all, if the liquidity-provision mechanism is
real).

**Implementation**: `src/regime.py::volatility_regime` bucketing SPY's own
trailing realized volatility (an external, broad-market proxy independent
of the 23-name traded universe) into tertiles, gating the risk-managed
reversal strategy from `docs/short_term_reversal.md` to be flat on days
outside the target bucket. Three gates were pre-specified before looking
at results — "high-vol tertile only," "exclude the low-vol tertile," and
"below the 200-day trend" — each tested at two different percentile
lookback windows (252 and 504 trading days) as a built-in robustness
check, plus the Markov-switching filtered probability as an independent,
differently-estimated cross-check on the same idea.

**Why test the same rule at two lookback windows on purpose**: a result
that only holds at one specific, arbitrarily-chosen window width is
indistinguishable from a rule that happened to fit the noise in that one
parameterization — exactly the failure mode `walk_forward_with_selection`
demonstrated for the reversal strategy's own lookback parameter (see
RESEARCH_MEMO.md §7.1). Requiring agreement across two honestly-different,
independently-reasonable choices is a cheap, non-exhaustive way to guard
against exactly that.

## 5. Empirical postscript — what this project actually found

The "high-vol tertile only" gate looked strong at a 504-day lookback
window (Sharpe 0.32, up from 0.23 ungated) and then **flipped to negative**
(Sharpe -0.06) at a 252-day lookback window — rejected as non-robust, a
direct instance of the failure mode this section's methodology was
designed to catch. A related but less aggressive rule — **exclude only the
lowest volatility tertile**, rather than require the highest — gave a
consistent, positive result at both lookback windows (Sharpe 0.42 and
0.44), and became this project's new headline result (Sharpe 0.44, PSR
0.91, max drawdown -16.8%, active 63% of days — see RESEARCH_MEMO.md
§7.1b, §7.4). The independent Markov-switching measure agreed with the
rule-based gates only 65% of the time and gave a weaker, though
directionally consistent, standalone result (Sharpe 0.13) — a reminder
that "regime" is a fuzzier, harder-to-pin-down concept than a single clean
number suggests, and that two imperfectly-agreeing methods pointing the
same direction is meaningfully better evidence than one method alone, but
still short of a tightly-confirmed result. The trend-based gates
underperformed the volatility-based ones, suggesting (contrary to a naive
first guess) that this strategy's edge is about volatility specifically,
not market direction.

## References

- Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle." *Econometrica*,
  57(2), 357-384.
- Ang, A., & Bekaert, G. (2002). "International Asset Allocation with
  Regime Shifts." *Review of Financial Studies*, 15(4), 1137-1187.
- Lehmann, B. N. (1990). "Fads, Martingales, and Market Efficiency."
  *Quarterly Journal of Economics*, 105(1), 1-28. (The liquidity-provision
  mechanism this section's hypothesis is derived from.)
