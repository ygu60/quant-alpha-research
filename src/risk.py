"""
Risk management overlay: transforms a strategy's raw target weights into
risk-controlled weights, without touching the alpha signal itself.

Design principle: every function here must only use information available
at the close of the day the weight is FOR, exactly like the strategies in
strategies.py -- backtest.py still does the single forward shift before
applying weights to next-day returns, so an overlay that is itself causal
keeps the whole pipeline look-ahead-free. Each function documents which
window of history it reads to prove that.

Four controls, applied in order (each is optional and composable):

1. volatility_target   -- scale the whole book up/down daily so trailing
   realized volatility tracks a target annual vol. This is the single most
   important small-account risk control: a strategy that "worked" at
   whatever volatility its raw signal happened to produce is not a risk
   management plan.
2. cap_position_weight  -- hard per-name cap (no single small/illiquid name
   can dominate the book, regardless of what the signal wants).
3. cap_gross_exposure   -- hard cap on total (long + short) leverage.
4. drawdown_kill_switch -- flattens the entire book once trailing drawdown
   breaches a threshold, and stays flat until the drawdown recovers to a
   shallower level (hysteresis, so it doesn't flap on the way back in).
   This is a sequential simulation (state depends on its own past output),
   documented inline for why it exactly reproduces backtest.py's return/cost
   math so running the returned weights back through run_backtest() gives
   numbers consistent with the halt decisions that were actually made.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def volatility_target(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    target_vol: float = 0.10,
    lookback: int = 20,
    max_leverage: float = 2.0,
    min_leverage: float = 0.2,
) -> pd.DataFrame:
    """Scale `weights` so the book's trailing realized annualized vol tracks
    `target_vol`.

    The scale applied to the weight decided AT close of day t is computed
    from a rolling window of returns realized ON OR BEFORE day t (i.e. the
    strategy's own day-over-day P&L using weights decided at t-1, t-2, ...),
    so it never uses information from day t+1 onward. Before `lookback`
    days of history exist, leverage defaults to 1.0 (no scaling) rather than
    an unstable early estimate.
    """
    asset_returns = prices.pct_change()
    applied = weights.shift(1).fillna(0.0)
    base_returns = (applied * asset_returns).sum(axis=1)

    trailing_vol = base_returns.rolling(lookback, min_periods=lookback).std(ddof=1) * np.sqrt(TRADING_DAYS)
    scale = (target_vol / trailing_vol).clip(lower=min_leverage, upper=max_leverage)
    scale = scale.fillna(1.0)

    return weights.mul(scale, axis=0)


def cap_position_weight(weights: pd.DataFrame, max_weight: float = 0.35) -> pd.DataFrame:
    """Clip every single-name weight to +/-max_weight. A pure risk cap, not
    a renormalization -- if the signal wants more concentration than this,
    the excess is simply not taken, which reduces (never increases) gross
    exposure.
    """
    return weights.clip(lower=-max_weight, upper=max_weight)


def cap_gross_exposure(weights: pd.DataFrame, max_gross: float = 1.5) -> pd.DataFrame:
    """Scale down (never up) any day whose total |long| + |short| exposure
    exceeds `max_gross`.
    """
    gross = weights.abs().sum(axis=1)
    scale = (max_gross / gross).clip(upper=1.0)
    scale = scale.replace([np.inf, -np.inf], 1.0).fillna(1.0)
    return weights.mul(scale, axis=0)


def drawdown_kill_switch(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float = 10.0,
    max_drawdown: float = 0.15,
    recovery_drawdown: float = 0.05,
) -> pd.DataFrame:
    """Flatten the book once the underlying signal's drawdown breaches
    `max_drawdown`, stay flat until it recovers to within `recovery_drawdown`
    of its peak.

    The drawdown gauge here is deliberately computed from the ORIGINAL,
    never-halted `weights` applied throughout -- i.e. "what would this
    strategy's own equity curve be doing right now if we always stayed
    invested" -- rather than from the halt-adjusted output this function
    produces. Two failure modes were found (via backtest, not by
    inspection) from more "obvious" designs and are why this one is used:

    1. Gauging recovery from the ACTUAL (halt-adjusted) equity curve: once
       flattened, that curve returns exactly 0% forever, so its own
       drawdown from its own peak can never shrink -- it permanently locks
       the strategy flat after the very first breach. Caught by backtesting
       real data, where a trigger in year 3 of a 9-year sample was never
       followed by a single re-entry.
    2. Gauging the TRIGGER from the actual curve while gauging RECOVERY from
       the shadow curve: after re-entering, the actual curve is still deep
       below its own PRE-CRASH peak (it only just started recovering), so
       it immediately re-breaches `max_drawdown` relative to that stale
       peak and flattens again one day later -- a flapping on/off pattern
       every time it tries to re-enter. Also caught by backtesting.

    Using one single, self-consistent gauge (the raw signal's own
    always-invested equity curve, decoupled from our own past halt
    decisions) avoids both: it has no memory of whether we personally acted
    on it, so there is no feedback loop between the halt decision and the
    thing the halt decision is measured against. Reproduces backtest.py's
    return/cost formula so it stays consistent with how the output would
    actually score. The decision for day t uses only returns realized
    through day t's close (from the weight decided at t-1) and is applied
    to the weight decided AT day t, which only affects the t -> t+1 return
    -- no leakage.
    """
    asset_returns = prices.pct_change()
    out = weights.copy()
    columns = weights.columns
    n = len(weights)

    equity, peak = 1.0, 1.0
    halted = False
    prev_applied = pd.Series(0.0, index=columns)  # raw weights two steps back, for turnover

    for i in range(1, n):
        applied = weights.iloc[i - 1]  # always the RAW weight -- decoupled from our own halt state
        ret = float((applied * asset_returns.iloc[i]).sum())
        turnover = float((applied - prev_applied).abs().sum())
        equity *= (1.0 + ret - turnover * (cost_bps / 10_000.0))
        peak = max(peak, equity)
        dd = equity / peak - 1.0

        if halted:
            if dd >= -recovery_drawdown:
                halted = False
        elif dd <= -max_drawdown:
            halted = True

        out.iloc[i] = 0.0 if halted else weights.iloc[i]
        prev_applied = applied

    return out


def apply_full_overlay(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    target_vol: float | None = 0.10,
    vol_lookback: int = 20,
    max_leverage: float = 2.0,
    min_leverage: float = 0.2,
    max_position: float | None = 0.35,
    max_gross: float | None = 1.5,
    max_drawdown: float | None = 0.15,
    recovery_drawdown: float = 0.05,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Compose the four controls in the documented order. Any control can be
    disabled by passing its threshold as None, so a strategy's raw and
    risk-managed versions can be compared apples-to-apples in the same
    report.
    """
    out = weights
    if target_vol is not None:
        out = volatility_target(prices, out, target_vol=target_vol, lookback=vol_lookback,
                                 max_leverage=max_leverage, min_leverage=min_leverage)
    if max_position is not None:
        out = cap_position_weight(out, max_weight=max_position)
    if max_gross is not None:
        out = cap_gross_exposure(out, max_gross=max_gross)
    if max_drawdown is not None:
        out = drawdown_kill_switch(prices, out, cost_bps=cost_bps, max_drawdown=max_drawdown,
                                    recovery_drawdown=recovery_drawdown)
    return out
