"""
Static matplotlib charts for the research memo. Palette and mark choices
follow the project's dataviz reference palette (fixed categorical hue
order, so color always identifies the same series across every chart --
"color follows the entity, never its rank"):

    reversal + risk overlay (existing best, pre-ML)  -> blue    #2a78d6
    SPY buy & hold                                    -> orange  #eb6834
    Universe equal-weight buy & hold                  -> aqua    #1baf7a
    ML strategy (this study)                          -> violet  #4a3aa7
    generic "positive/negative" diverging bars         -> blue / red (#2a78d6 / #e34948)

All charts render on the light chart surface (#fcfcfb) with muted gridlines
(#e1e0d9) and ink colors from the same reference palette, and use direct
end-of-line labels rather than relying on a legend box alone (mitigates the
palette's documented sub-3:1-contrast slots on the light surface).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

COLOR_REVERSAL = "#2a78d6"   # categorical slot 1, blue
COLOR_SPY = "#eb6834"        # slot 2, orange
COLOR_UNIVERSE_BH = "#1baf7a"  # slot 3, aqua
COLOR_REGIME = "#eda100"    # slot 4, yellow -- direct end-label required (low light-surface contrast)
COLOR_ML = "#4a3aa7"         # slot 7, violet
COLOR_POSITIVE = "#2a78d6"   # diverging pole (blue)
COLOR_NEGATIVE = "#e34948"   # diverging pole (red)

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.title.set_color(INK_PRIMARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def _end_label(ax, x, y, text, color):
    ax.annotate(text, xy=(x, y), xytext=(6, 0), textcoords="offset points",
                va="center", ha="left", fontsize=9.5, color=color, fontweight="bold")


def _end_labels_decluttered(ax, entries: list[tuple[float, float, str, str]], min_gap_frac: float = 0.05) -> None:
    """Place end-of-line labels at a fixed x (just past the right edge, in
    axes-fraction coordinates) with y positions decluttered in DATA
    coordinates so labels for series whose lines end close together don't
    overlap -- e.g. two strategies both ending near a 0% drawdown. Each
    label keeps a thin leader line back to its series' true end point when
    the label had to move.

    `entries`: list of (x_end, y_end, text, color).
    """
    if not entries:
        return
    ylo, yhi = ax.get_ylim()
    min_gap = (yhi - ylo) * min_gap_frac

    ordered = sorted(range(len(entries)), key=lambda i: entries[i][1])
    adjusted = [entries[i][1] for i in ordered]
    for k in range(1, len(adjusted)):
        if adjusted[k] - adjusted[k - 1] < min_gap:
            adjusted[k] = adjusted[k - 1] + min_gap

    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for k, i in enumerate(ordered):
        x_end, y_end, text, color = entries[i]
        y_label = adjusted[k]
        ax.annotate(text, xy=(1.0, y_label), xycoords=trans, xytext=(6, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=9.5, color=color, fontweight="bold")
        if abs(y_label - y_end) > min_gap * 0.2:
            # Vertical stub at the line's own true x (never introduce a
            # second, differently-typed x value here -- mixing a raw
            # ax.get_xlim() float with a datetime x_end previously corrupted
            # the whole axis's date scaling).
            ax.plot([x_end, x_end], [y_end, y_label], color=color,
                    linewidth=0.6, linestyle=":", alpha=0.6, zorder=4, clip_on=False)


def plot_equity_curves(series_dict: dict[str, pd.Series], title: str, out_path: Path) -> None:
    """series_dict: label -> daily return Series (not yet cumulated). All
    series are reindexed to their common date range before cumulating so
    the comparison is apples-to-apples over identical OOS dates.
    """
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    common = None
    for s in series_dict.values():
        common = s.index if common is None else common.intersection(s.index)
    label_entries = []
    for label, returns in series_dict.items():
        color = _COLOR_MAP.get(label, INK_SECONDARY)
        equity = (1 + returns.loc[common]).cumprod()
        ax.plot(equity.index, equity.values, color=color, linewidth=1.8, zorder=3)
        label_entries.append((equity.index[-1], equity.values[-1], label, color))
    ax.axhline(1.0, color=BASELINE, linewidth=1.0, linestyle="--", zorder=1)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel("Growth of $1")
    _style_axes(ax)
    _end_labels_decluttered(ax, label_entries)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_drawdown(series_dict: dict[str, pd.Series], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    common = None
    for s in series_dict.values():
        common = s.index if common is None else common.intersection(s.index)
    label_entries = []
    for label, returns in series_dict.items():
        color = _COLOR_MAP.get(label, INK_SECONDARY)
        equity = (1 + returns.loc[common]).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        ax.fill_between(drawdown.index, drawdown.values, 0, color=color, alpha=0.18, zorder=2)
        ax.plot(drawdown.index, drawdown.values, color=color, linewidth=1.4, zorder=3)
        label_entries.append((drawdown.index[-1], drawdown.values[-1], label, color))
    ax.axhline(0.0, color=BASELINE, linewidth=1.0, zorder=1)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel("Drawdown from peak")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _style_axes(ax)
    _end_labels_decluttered(ax, label_entries)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_rolling_sharpe(series_dict: dict[str, pd.Series], window: int, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    common = None
    for s in series_dict.values():
        common = s.index if common is None else common.intersection(s.index)
    label_entries = []
    for label, returns in series_dict.items():
        color = _COLOR_MAP.get(label, INK_SECONDARY)
        r = returns.loc[common]
        rolling_sharpe = (r.rolling(window).mean() / r.rolling(window).std(ddof=1)) * np.sqrt(252)
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, color=color, linewidth=1.5, zorder=3)
        valid = rolling_sharpe.dropna()
        if len(valid):
            label_entries.append((valid.index[-1], valid.values[-1], label, color))
    ax.axhline(0.0, color=BASELINE, linewidth=1.0, zorder=1)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel(f"Rolling {window}-day annualized Sharpe")
    _style_axes(ax)
    _end_labels_decluttered(ax, label_entries)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_feature_importance(importances: pd.Series, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ordered = importances.sort_values(ascending=True)
    ax.barh(ordered.index, ordered.values, color=COLOR_ML, zorder=3, height=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("Mean importance across walk-forward windows")
    _style_axes(ax)
    ax.grid(True, axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_window_sharpe_bars(window_stats: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    sharpes = window_stats["oos_sharpe"].to_numpy()
    colors = [COLOR_POSITIVE if s >= 0 else COLOR_NEGATIVE for s in sharpes]
    x = np.arange(len(sharpes))
    ax.bar(x, sharpes, color=colors, width=0.7, zorder=3)
    ax.axhline(0.0, color=BASELINE, linewidth=1.0, zorder=1)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel("Out-of-sample Sharpe (per window)")
    ax.set_xlabel("Walk-forward window (chronological)")
    ax.set_xticks([])
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_permutation_test(null_sharpes: np.ndarray, real_sharpe: float, title: str, out_path: Path) -> None:
    """Histogram of Sharpe ratios from shuffled-label negative-control runs,
    with the real (unshuffled) model's Sharpe marked -- the visual version
    of "is this result distinguishable from noise".
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.hist(null_sharpes, bins=max(5, len(null_sharpes) // 2), color=INK_MUTED, alpha=0.55,
            edgecolor=SURFACE, zorder=2)
    ax.axvline(real_sharpe, color=COLOR_ML, linewidth=2.2, zorder=3)
    ax.annotate(f"real model: {real_sharpe:.2f}", xy=(real_sharpe, ax.get_ylim()[1] * 0.9),
                xytext=(8, 0), textcoords="offset points", color=COLOR_ML, fontweight="bold", fontsize=9.5)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("Out-of-sample Sharpe ratio")
    ax.set_ylabel("Shuffled-label draws")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


_COLOR_MAP = {
    "Reversal + risk overlay": COLOR_REVERSAL,
    "Reversal + risk overlay + regime gate": COLOR_REGIME,
    "SPY buy & hold": COLOR_SPY,
    "Universe buy & hold": COLOR_UNIVERSE_BH,
    "ML strategy (risk-managed)": COLOR_ML,
    "ML strategy (raw)": COLOR_ML,
}
