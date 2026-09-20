# Deep Learning for Return Prediction

## 1. Motivation

`docs/ml_signal_search.md` covers the general supervised-learning
formalization of a trading signal and the purging/permutation-test
methodology used to validate one honestly. This chapter is narrower: given
that a linear model (Ridge) and two tree ensembles (Random Forest,
gradient boosting) were already tried on the same feature set and came up
empty (RESEARCH_MEMO.md §10.1-10.2), does a neural network — strictly more
expressive, capable in principle of approximating any continuous function
of the inputs — find something they missed?

This is worth asking on its own terms rather than assumed away, because
"more capacity" genuinely helps in some domains (large-scale image and
language modeling, where the training set is enormous relative to the
input dimensionality) and genuinely hurts in others (small, noisy tabular
data, where a flexible model has more ways to fit sampling noise rather
than signal). Which regime a given problem is in is an empirical question.
Financial return prediction on a modest universe over a modest history —
this project's 23 names, roughly 9 years of daily data — has long been
argued to sit closer to the second case (see e.g. the general finding
across the empirical asset-pricing literature that simple/regularized
models tend to be highly competitive with complex ones on tabular
firm-characteristic data), but "argued" is not the same as "tested here."

## 2. What a neural network adds, formally

A feedforward network (multilayer perceptron, MLP) with one hidden layer
computes

$$
\hat{y} = W_2 \, \sigma(W_1 x + b_1) + b_2
$$

where $x \in \mathbb{R}^d$ is the feature vector, $\sigma(\cdot)$ is an
elementwise nonlinearity (ReLU, $\sigma(z) = \max(0, z)$, in this
project's implementation), and $W_1, b_1, W_2, b_2$ are learned
parameters. The universal approximation theorem (Cybenko, 1989; Hornik,
1991) guarantees that, with enough hidden units, a network of this form
can approximate any continuous function on a compact domain arbitrarily
well. This is a statement about REPRESENTATIONAL capacity, not about
whether a network trained on a FINITE, NOISY sample will actually recover
the true function rather than the sample's idiosyncratic noise — the
latter is a statistical estimation question, governed by the bias-variance
tradeoff, not resolved by the approximation theorem at all. A model class
rich enough to fit anything is also rich enough to fit noise; whether it
does so in practice depends on the ratio of usable signal to sample size,
and on how aggressively the fitting procedure is regularized.

Every design choice in `src/dl.py::TorchMLP` is a regularization decision,
not a modeling flourish:

- **Depth/width kept small** (1-2 hidden layers, 16-32 units): fewer
  parameters relative to a training set of a few thousand rows per
  walk-forward window.
- **Dropout** (randomly zeroing a fraction of hidden activations during
  training, Srivastava et al. 2014): approximates training an ensemble of
  thinned sub-networks and averaging them, empirically one of the more
  effective regularizers for this failure mode.
- **Weight decay** ($L_2$ penalty on the parameters, added to the loss as
  $\lambda \lVert \theta \rVert_2^2$): shrinks parameters toward zero,
  the neural-network analogue of ridge regression's regularization.
- **Early stopping on a held-out validation slice**: training is halted
  once validation loss stops improving for `patience` epochs, using the
  chronologically LAST slice of the training data as validation (never the
  test data) — this is itself a form of regularization (Prechelt, 1998,
  shows early stopping is approximately equivalent to an explicit
  parameter-norm penalty under reasonable conditions), on top of being the
  mechanism that decides how long to train at all.

## 3. Why the SAME validation loop as every other model matters

`src/dl.py::TorchMLP` is deliberately built as a class with `.fit(X, y)`
and `.predict(X)` methods matching the scikit-learn estimator interface,
so it plugs into `src/ml.py::run_ml_walk_forward` — the same purged
walk-forward loop, the same `shuffle_labels` negative control, the same
cross-sectional ranking into a long-only book — used for Ridge, Random
Forest, and gradient boosting. This is a methodological choice, not just a
code-reuse convenience: a common failure mode in less careful comparisons
is giving a new, more complex model a MORE lenient validation procedure
(more hyperparameter search, a friendlier train/test split, a metric
computed slightly differently) than the baselines it's being compared
against, which mechanically inflates the complex model's apparent
advantage regardless of whether one exists. Using literally the same
function, with only `build_model` swapped out, rules that out by
construction.

## 4. What the hypothesis predicts, and how it was tested

**Falsifiable prediction:** if the reason the linear/tree-based models in
`docs/ml_signal_search.md` found no edge is that the true relationship
between these 15 features and forward returns is nonlinear in a way trees
can't easily represent (trees partition the feature space into axis-
aligned rectangles; a network can represent smooth, curved decision
surfaces more naturally), a neural network should show a materially higher
out-of-sample Sharpe and a permutation-test result further from its own
noise floor.

**Implementation**: a small, PRE-REGISTERED $2\times2$ grid (hidden-layer
configuration $\times$ dropout strength — not searched for a better-
looking result after the fact, for the same reason `walk_forward_with_selection`
found searching hurt the reversal strategy: every additional configuration
tried is another chance to find one that looks good on this one sample by
chance). Trained on the identical 15-feature, 10-day-horizon setup as the
best tree-based result, scored through the identical purged walk-forward
and permutation test.

## 5. Empirical postscript — what this project actually found

The best of four configurations (2 hidden layers of 32 and 16 units,
dropout 0.4) reached Sharpe 0.15 — comparable to or WEAKER than the tree-
based Random Forest result (0.35) at the same horizon, not better. A
permutation test put it at only the 75th percentile of its own shuffled-
label noise floor (weaker evidence than the tree-based result's already-
unconvincing 80th percentile). Applying the identical risk overlay used
throughout this project made it substantially worse, not better: Sharpe
-0.18 with a -50.8% max drawdown, against SPY's 0.85 Sharpe and -33.7%
drawdown over the identical dates — the single worst risk-managed result
of any strategy tried anywhere in this project.

The falsifiable prediction in §4 is falsified: more representational
capacity did not recover a signal the simpler models missed. Read together
with §2's framing, the most likely explanation is not that neural networks
are unsuited to financial prediction in general (they are used
successfully elsewhere in the industry, typically with far larger
training sets, richer data such as order-book or alternative data, and
extensive infrastructure this project does not have), but that the
bottleneck here is the signal-to-noise ratio available in 9 years of
daily OHLCV data on a 23-name universe specifically — a data and sample-
size constraint that no amount of additional model capacity can fix, and
that made a genuinely more flexible model find MORE ways to fit noise, not
fewer.

## References

- Cybenko, G. (1989). "Approximation by Superpositions of a Sigmoidal
  Function." *Mathematics of Control, Signals and Systems*, 2(4), 303-314.
- Hornik, K. (1991). "Approximation Capabilities of Multilayer Feedforward
  Networks." *Neural Networks*, 4(2), 251-257.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov,
  R. (2014). "Dropout: A Simple Way to Prevent Neural Networks from
  Overfitting." *Journal of Machine Learning Research*, 15(1), 1929-1958.
- Prechelt, L. (1998). "Early Stopping — But When?" in *Neural Networks:
  Tricks of the Trade*, Springer Lecture Notes in Computer Science, vol.
  1524.
- Gu, S., Kelly, B., & Xiu, D. (2020). "Empirical Asset Pricing via Machine
  Learning." *Review of Financial Studies*, 33(5), 2223-2273. (Large-scale
  comparison of ML methods, including neural networks, for return
  prediction on the broad U.S. cross-section — useful context for what
  this kind of approach looks like at institutional data scale, a very
  different regime from this project's 23-name universe.)
