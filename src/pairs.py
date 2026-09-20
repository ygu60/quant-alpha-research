"""
Pair selection for the pairs-trading strategy, following the distance
method of Gatev, Goetzmann & Rouwenhorst (2006), "Pairs Trading: Performance
of a Relative-Value Arbitrage Rule" (Review of Financial Studies).

Why this module exists: the original implementation hardcoded a single pair
(SIM1/SIM2 on synthetic data). On a real universe, hand-picking "the pair
that looks cointegrated" AFTER seeing how it trades is a textbook overfitting
mistake -- you are selecting on the outcome you're about to report. Gatev et
al.'s fix, which we replicate here, is to select pairs using only a prior
FORMATION period (normalized price distance, no trading signal involved),
then trade the selected pairs in a separate, later TRADING period. The
formation step never looks at trading-period returns.

Candidates are restricted to the same sector (data.UNIVERSE_GROUPS) rather
than the whole cross-section: two economically unrelated small caps having a
low price distance over one formation window is far more likely to be
coincidence than two regional banks moving together on shared rate/credit
exposure. Restricting the search space also shrinks the multiple-comparisons
problem (fewer candidate pairs tested = less chance of a spurious best pair).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .strategies import zscore_pairs


def _normalize(prices: pd.Series) -> pd.Series:
    """Rebase a price series to start at 1.0 so distance is scale-free."""
    valid = prices.dropna()
    if len(valid) == 0:
        return prices
    return prices / valid.iloc[0]


def select_pairs_by_distance(
    formation_prices: pd.DataFrame,
    candidates: list[str],
    top_n: int = 2,
    min_obs: int = 60,
) -> list[tuple[str, str]]:
    """Rank all within-`candidates` pairs by sum-of-squared-distance of their
    normalized price paths over `formation_prices`, and return the `top_n`
    closest pairs. This is a FORMATION-period-only computation -- it must
    never be called on data that includes the period you intend to trade.
    """
    scored = []
    for a, b in combinations(candidates, 2):
        if a not in formation_prices.columns or b not in formation_prices.columns:
            continue
        pa = _normalize(formation_prices[a])
        pb = _normalize(formation_prices[b])
        joint = pd.concat([pa, pb], axis=1).dropna()
        if len(joint) < min_obs:
            continue
        distance = float(((joint.iloc[:, 0] - joint.iloc[:, 1]) ** 2).sum())
        scored.append((distance, a, b))

    scored.sort(key=lambda x: x[0])
    return [(a, b) for _, a, b in scored[:top_n]]


def adaptive_pairs_weights(
    prices: pd.DataFrame,
    train_days: int,
    candidates: list[str],
    top_n: int = 2,
    lookback: int = 20,
    entry_z: float = 1.5,
    exit_z: float = 0.3,
) -> pd.DataFrame:
    """Select pairs from `candidates` using only `prices.iloc[:train_days]`
    (the formation window), then generate combined zscore_pairs weights for
    the selected pairs across the FULL `prices` index.

    This is designed to be dropped straight into `walk_forward()` as a
    `weight_fn`: walk_forward already only scores the post-`train_days`
    portion of the returned weights, so the trading signal itself (like all
    other strategies here) still only ever uses information up to each day's
    close -- the one extra thing this function must get right, and does, is
    that WHICH pairs get selected is decided before the test window starts.
    """
    formation = prices.iloc[:train_days]
    pairs = select_pairs_by_distance(formation, candidates, top_n=top_n)

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if not pairs:
        return weights

    per_pair_weight = 1.0 / len(pairs)
    for a, b in pairs:
        pair_w = zscore_pairs(prices, a, b, lookback=lookback, entry_z=entry_z, exit_z=exit_z)
        weights[a] = weights[a] + per_pair_weight * pair_w[a]
        weights[b] = weights[b] + per_pair_weight * pair_w[b]

    return weights
