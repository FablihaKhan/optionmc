"""Figures for the protective put optimizer.

Kept separate from `plots.py` so the twelve figures the original scope requires
are untouched. The palette, the style and the save helper are imported from
there, so the two sets cannot drift apart: blue is still what this project
computes, orange the benchmark, grey a reference that is not a series.

Two encodings are used repeatedly here. Confidence levels 95% and 99% are
*ordered*, so they get a single-hue light-to-dark ramp rather than two hues.
VaR and CVaR are *different measures*, so they are separated by line style --
which also means neither figure depends on colour alone to be read.
"""
import matplotlib.pyplot as plt
import numpy as np

from .plots import (BENCHMARK, LSMC, MARKET, ORDINAL_3, REFERENCE, _finish,
                    dollars, percent)

# 95% and 99% as an ordered pair: light then dark, same hue.
LEVEL_RAMP = (ORDINAL_3[0], ORDINAL_3[2])


def plot_protection_cost_frontier(cost_percent, protection, strikes, moneyness,
                                  pareto, highlight=None, highlight_label=None,
                                  title="Protection against cost",
                                  level_label="99% CVaR"):
    """Every candidate put as one point: what it costs against what it removes.

    Up is more tail protection, right is more expensive. A candidate that some
    other candidate beats on both counts is drawn hollow and left off the
    frontier line -- a dominated hedge is never presented as a choice.
    """
    cost_percent = np.asarray(cost_percent, dtype=float)
    protection = np.asarray(protection, dtype=float)
    pareto = np.asarray(pareto, dtype=bool)

    fig, ax = plt.subplots(figsize=(9, 6))

    order = np.argsort(cost_percent)
    front = order[pareto[order]]
    if front.size > 1:
        ax.plot(cost_percent[front], protection[front], color=LSMC,
                linewidth=2.0, zorder=2, label="Pareto frontier")

    if np.any(pareto):
        ax.scatter(cost_percent[pareto], protection[pareto], s=110, color=LSMC,
                   edgecolor="white", linewidth=2.0, zorder=4,
                   label="Pareto efficient")
    if np.any(~pareto):
        ax.scatter(cost_percent[~pareto], protection[~pareto], s=110,
                   facecolor="white", edgecolor=REFERENCE, linewidth=1.8,
                   zorder=3, label="dominated")

    if highlight is not None:
        ax.scatter(cost_percent[highlight], protection[highlight], s=320,
                   facecolor="none", edgecolor=BENCHMARK, linewidth=2.4,
                   zorder=5, label=highlight_label or "selected")

    for x, y, k, m in zip(cost_percent, protection, strikes, moneyness):
        ax.annotate(f"K={k:g}\n{m:.1%}", xy=(x, y), xytext=(0, 14),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=REFERENCE)

    span = protection.max() - protection.min()
    ax.set_ylim(protection.min() - 0.12 * span - 1, protection.max() + 0.30 * span + 1)
    ax.xaxis.set_major_formatter(percent(2))
    ax.yaxis.set_major_formatter(percent(0))
    return fig, _finish(ax, title,
                        "hedge cost [% of the share position]",
                        f"{level_label} reduction [%]")


def plot_efficiency_by_strike(strikes, saved_95, saved_99,
                              title="Tail loss avoided per dollar of premium"):
    """How many dollars of CVaR each dollar of premium buys, strike by strike."""
    strikes = np.asarray(strikes, dtype=float)
    x = np.arange(strikes.size, dtype=float)
    width, gap = 0.36, 0.02

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars_95 = ax.bar(x - width / 2 - gap, saved_95, width, color=LEVEL_RAMP[0],
                     label="95% CVaR")
    bars_99 = ax.bar(x + width / 2 + gap, saved_99, width, color=LEVEL_RAMP[1],
                     label="99% CVaR")

    for bars in (bars_95, bars_99):
        for bar in bars:
            ax.annotate(f"${bar.get_height():,.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, color=REFERENCE)

    ax.axhline(1.0, color=REFERENCE, linestyle="--", linewidth=1.2)
    # Dollar signs are escaped: an unescaped pair puts matplotlib into mathtext
    # and the sentence comes out as run-together italics.
    # Every bar clears the break-even line, so the caption has to sit over the
    # bars; a white plate keeps it readable instead of half-hidden behind one.
    ax.annotate(r"break-even: \$1 of tail loss avoided per \$1 spent",
                xy=(x[0] - 0.45, 1.0), xytext=(0, 6),
                textcoords="offset points", fontsize=9, color=REFERENCE,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.88,
                          pad=2.0))

    ax.set_xticks(x)
    ax.set_xticklabels([f"{k:g}" for k in strikes])
    ax.set_ylim(top=max(max(saved_95), max(saved_99), 1.0) * 1.20)
    ax.yaxis.set_major_formatter(dollars(2))
    return fig, _finish(ax, title, "strike [$]",
                        "CVaR avoided per $1 of premium")


def plot_reduction_by_moneyness(moneyness, reductions,
                                title="Tail-risk reduction against strike"):
    """Reduction in each risk measure as the hedge moves toward the money.

    `reductions` maps a label like "99% CVaR" to its series. VaR is dashed and
    CVaR solid, so the two measures stay apart for a reader who cannot separate
    the two blues.
    """
    moneyness = np.asarray(moneyness, dtype=float) * 100.0
    fig, ax = plt.subplots(figsize=(9, 5.5))

    styles = {
        "95% VaR": (LEVEL_RAMP[0], "--"),
        "99% VaR": (LEVEL_RAMP[1], "--"),
        "95% CVaR": (LEVEL_RAMP[0], "-"),
        "99% CVaR": (LEVEL_RAMP[1], "-"),
    }
    for label, values in reductions.items():
        colour, style = styles.get(label, (MARKET, "-"))
        ax.plot(moneyness, values, color=colour, linestyle=style,
                marker="o", markersize=6, linewidth=2.0, label=label)
        ax.annotate(label, xy=(moneyness[-1], values[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=9, color=colour,
                    va="center")

    ax.set_xlim(moneyness.min() - 1.0, moneyness.max() + 3.0)
    ax.xaxis.set_major_formatter(percent(1))
    ax.yaxis.set_major_formatter(percent(0))
    # A long handle so the dash pattern -- which is what separates VaR from
    # CVaR -- is actually visible in the legend and not just on the lines.
    ax.legend(handlelength=3.4, loc="upper left")
    return fig, _finish(ax, title, "strike as a percentage of spot",
                        "reduction against the unhedged portfolio [%]",
                        legend=False)
