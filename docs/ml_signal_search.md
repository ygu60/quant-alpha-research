# Machine-Learning Signal Search: Formalism, Leakage, and the Permutation Test

## 1. Motivation: searching for structure without an economic prior

Every other strategy in this project starts from a specific economic
mechanism (bid-ask bounce, a mean-reverting spread, a calendar liquidity
flow) and only then asks whether the data supports it. This chapter is the
opposite exercise: give a flexible statistical model a broad menu of
generic, "non-semantic" price/volume statistics — features chosen for no
reason other than "commonly used in technical/quantitative analysis," with
no specific economic story attached to any one of them — and let the model
search for whatever cross-sectional or time-series structure best predicts
forward returns, if any exists.

This is a legitimate complement to theory-first research, not a
replacement for it: a real anomaly could in principle exist that isn't
captured by any of the classical signals in the literature. But it comes
with a much weaker prior. A signal derived from theory has decades of
out-of-sample replication (across different markets, different eras,
different research teams) behind it before this project ever tests it. A
pattern a flexible model finds in one specific 9-year sample of 23 stocks
has *no* such backing — it could be the real thing, or it could be the
model fitting noise that happens to look like structure in this
particular sample. That asymmetry is why this chapter's validation
machinery (purging, and especially the permutation test in §4) has to do
more work than the validation for a theory-motivated signal: there is no
prior evidence to fall back on if the out-of-sample test is ambiguous.

## 2. Formalizing "a trading signal" as a supervised-learning problem

Let $X_{i,t} \in \mathbb{R}^p$ be a vector of $p$ features observed for
asset $i$ at the close of day $t$ (`src/features.py::compute_features`
produces $p=15$: momentum at several horizons, realized volatility,
RSI, a moving-average gap, a Bollinger z-score, rolling skew and
lag-1 autocorrelation, a volume-surge ratio, and same-day cross-sectional
percentile ranks of three of those — see that module for the exact list).
Define the forward $h$-day return as the supervised-learning **label**:

$$
y_{i,t} = \frac{P_{i,t+h}}{P_{i,t}} - 1
$$

(`src/features.py::forward_return`). A trading signal is then a function
$f_\theta: \mathbb{R}^p \to \mathbb{R}$, fit by minimizing a loss (squared
error, for the regressors used here) over training examples
$\{(X_{i,t}, y_{i,t})\}$:

$$
\hat\theta = \arg\min_\theta \sum_{(i,t) \in \text{train}} \big(y_{i,t} - f_\theta(X_{i,t})\big)^2
$$

and the resulting scores $\hat y_{i,t} = f_{\hat\theta}(X_{i,t})$ are used
purely as a **cross-sectional ranking device**: on each day, sort assets by
$\hat y_{i,t}$ and hold the top quintile, equal-weighted, long-only
(`src/ml.py::run_ml_walk_forward`). What matters for a profitable trading
rule is not that $f_{\hat\theta}$ accurately predicts the *level* of
$y_{i,t}$ — a notoriously close-to-impossible task for daily equity
returns — only that it produces a *ranking* correlated with the true
ranking of forward returns, since the strategy only ever acts on relative
order, never on the predicted magnitude.

Three model classes are used, in increasing order of flexibility: Ridge
regression (a linear baseline, $f_\theta(X) = \theta^\top X$ with an
$L_2$ penalty), Random Forest, and histogram-based Gradient Boosting (both
non-parametric, tree-ensemble regressors capable of fitting nonlinear
interactions the linear model cannot). All three are run with fixed,
deliberately conservative hyperparameters (shallow trees, large
min-leaf-size, meaningful regularization) chosen *before* looking at any
result — a train-only nested search over a small grid was tried for the
much simpler short-term reversal strategy (`docs/short_term_reversal.md`;
RESEARCH_MEMO.md §7.1) and made out-of-sample performance *worse*, which
is exactly the mechanism §5 warns against and there was no reason to
expect a higher-capacity model would be immune to it.

## 3. Why overlapping labels require purging: a leakage argument

Every other strategy's walk-forward validation (`src/walk_forward.py`)
only ever needs "compute the signal using data through day $t$, apply it
to day $t \to t+1$'s return" — a signal computed at $t$ cannot see
anything after $t$, by construction. The ML label $y_{i,t}$ breaks this
cleanly: it is defined using $P_{i,t+h}$, a price $h$ days in the
*future*. This is exactly as intended — it's the supervised-learning
target, not a feature — but it creates a subtle leakage path at the
train/test boundary of a walk-forward window that a hand-built signal
never has to worry about.

Consider a training window ending at day $T$ (the last day of the training
slice) and a test window beginning at day $T+1$. A training example from
day $T - h + 1$ has label

$$
y_{i,\,T-h+1} = \frac{P_{i,\,T+1}}{P_{i,\,T-h+1}} - 1,
$$

which depends on $P_{i,T+1}$ — a price that falls *inside* the test
period. More generally, any training example from days $T-h+1, \dots, T$
has a label window that overlaps the test period. If the model is fit
using these examples, it has effectively seen (through the label, not
through a feature — but the distinction doesn't help, since the label is
exactly what training minimizes error against) information about
test-period prices before ever making a test-period prediction. This is a
covariance-leakage argument, not merely a suspicious-looking coincidence:
$\text{Cov}(y_{i,T-h+1}, y_{j,T+1-h'})$ for a nearby test-period label
$y_{j, T+1-h'}$ is generically nonzero whenever their underlying price
windows overlap, so a model that fits well to the leaking training labels
can show spuriously good performance on nearby test labels purely through
this shared-price-window channel, with no genuine predictive relationship
between $X$ and future $y$ at all.

The fix, standard in the financial-ML literature (**Lopez de Prado, 2018,
*Advances in Financial Machine Learning*, ch. 7**, calls this "purging"),
is simple once identified: drop every training example whose label window
extends past the train/test boundary. `src/ml.py::run_ml_walk_forward`
does this by truncating the training slice to `train_dates[:-horizon]`
before fitting — the last $h$ days of the nominal training window
contribute no training examples at all. This is regression-tested in
`tests/test_ml.py::test_purge_prevents_training_label_leakage` by an
adversarial construction: shock *only* test-period prices by a large
factor and confirm a deterministic model (Ridge) fits byte-identical
training labels regardless of the shock — if the purge boundary were off
by even a few days, this test would fail.

## 4. The permutation test: a Monte Carlo null distribution for the Sharpe ratio

Purging prevents *leakage*, but it does not by itself prove a model found
a *real* relationship rather than noise that happens to look tradeable in
this one sample — with enough features and a flexible enough model, some
amount of in-sample-flavored overfitting to the training data's noise can
still leak through into an out-of-sample Sharpe that is positive purely by
chance, especially given the walk-forward windows are not fully
independent of each other (adjacent windows' training data overlaps; see
RESEARCH_MEMO.md §9).

The **permutation test** addresses this directly, by simulation rather
than by a closed-form correction. Under the **null hypothesis** that there
is no real relationship between $X_{i,t}$ and $y_{i,t}$, the labels are
*exchangeable* with respect to the features — permuting them at random
should destroy any predictive relationship while preserving every other
statistical property of the label distribution (its marginal mean,
variance, skew, autocorrelation structure across assets on a given day,
etc.). So: refit the exact same model, on the exact same features, but
with the **training labels randomly shuffled** within each window
(`src/ml.py::run_ml_walk_forward(..., shuffle_labels=True)`), and record
the resulting out-of-sample Sharpe ratio. Repeating this many times
(different random permutations) builds an empirical **null distribution**
for "what Sharpe ratio does this exact pipeline produce when there is, by
construction, no real signal to find" — a Monte Carlo estimate of the
sampling distribution of the test statistic (the walk-forward Sharpe)
under the null, in the same spirit as a bootstrap or randomization test in
classical statistics, just applied to a backtested trading rule instead of
a simple mean-difference statistic.

The real (unshuffled) model's Sharpe ratio is then compared against this
null distribution: if it falls well outside the bulk of the null draws
(conventionally, above roughly the 95th percentile for a one-sided test at
the 5% level), that is evidence the real labels contain more exploitable
structure than random noise does under this exact pipeline. If it sits
comfortably within the null distribution's range, the result is
**statistically indistinguishable from what the pipeline produces on pure
noise**, regardless of how attractive the raw Sharpe number looks in
isolation.

## 5. What the hypothesis predicts, and how it was tested

**Falsifiable prediction:** at least one of {Ridge, Random Forest,
Gradient Boosting} $\times$ {5-day, 10-day forecast horizon}, trained on
the 15-feature non-semantic set, should produce a walk-forward
out-of-sample Sharpe ratio that (a) clears a high percentile of its own
shuffled-label noise floor, and (b) survives the same risk-management
overlay used for the reversal strategy while still outperforming SPY
buy-and-hold or matching its stability — the explicit bar this
exploration was scoped against.

**Implementation**: `src/ml.py::run_ml_walk_forward`, same 252-day
train / 63-day test walk-forward cadence as every other strategy, purged
per §3, three models at horizon $h=5$, the best of those carried forward
to $h=10$ as a robustness check (not a search for the best horizon — a
single pre-specified follow-up check), then a 5-draw permutation test per
§4 on that configuration, then the identical risk overlay from
`docs/short_term_reversal.md` §4 applied without re-tuning.

## 6. Empirical postscript — what this project actually found

None of the six {model $\times$ horizon} combinations tested cleared
either bar. The best raw result (Random Forest, 10-day horizon) showed
Sharpe 0.35 (PSR 0.85) — but its permutation test placed the real result
at only the **80th percentile** of its own shuffled-label noise
distribution (`plots/permutation_test.png`), well short of the
95th-percentile-or-better standard §4 sets for calling a result
distinguishable from noise. Applying the identical risk overlay used for
short-term reversal turned that raw result into Sharpe **-0.08** with a
**-41.3%** max drawdown, against SPY buy-and-hold's Sharpe **0.85** and
**-33.7%** drawdown over the identical out-of-sample dates
(`plots/equity_curves.png`, `plots/drawdown.png` make the comparison
immediate). **No configuration tested is recommended for deployment.**
This is reported as a negative result rather than discarded: it rules out
this specific 15-feature, 3-model, 2-horizon search space on this
23-name, 9-year universe, and the purging + permutation-test scaffolding
built to test it is reusable for any future signal-search attempt on this
project without repeating the design work. See RESEARCH_MEMO.md §10 for
the full numbers and a discussion of why this particular search may have
come up empty (small cross-section, low-information-density daily OHLCV
data, modest sample size relative to model capacity).

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*.
  Wiley. Ch. 7 ("Cross-Validation in Finance") covers purging and
  embargoing in detail; the permutation-test logic in §4 is a close
  relative of the resampling-based significance tests discussed throughout
  that book.
- Bailey, D. H., & Lopez de Prado, M. (2012). "The Sharpe Ratio Efficient
  Frontier." *Journal of Risk*, 15(2), 3-44. (The Probabilistic Sharpe
  Ratio used elsewhere in this project as a complementary small-sample
  correction — see RESEARCH_MEMO.md §6 — addresses a related but distinct
  problem: how much to trust a *given* Sharpe ratio's magnitude, as
  opposed to the permutation test's question of whether the pipeline that
  produced it behaves differently from a pipeline with no real signal.)
- Good, P. I. (2005). *Permutation, Parametric, and Bootstrap Tests of
  Hypotheses* (3rd ed.). Springer. (Standard reference for the general
  permutation/randomization-test logic applied in §4 to a backtested
  Sharpe ratio rather than a classical test statistic.)
