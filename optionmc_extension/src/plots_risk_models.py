"""Figures comparing the GBM and historical-bootstrap risk engines.

Colour keeps its meaning from `plots.py`. Two assignments matter here:

    blue    the GBM engine -- the parametric model the project already had
    aqua    the historical sample and the bootstrap built from it, because
            that is what the market actually did rather than what a model says

and in the stress figures, matching figure 10 of the original set,

    blue    SPY only
    orange  SPY plus the put

Nothing is distinguished by colour alone: every series is labelled, the stress
chart is also separated by line style, and each figure ships its CSV.
"""
import matplotlib.pyplot as plt
import numpy as np

from .plots import (BENCHMARK, LSMC, MARKET, REFERENCE, _finish, dollars,
                    percent)

GBM_COLOUR = LSMC
EMPIRICAL_COLOUR = MARKET


def plot_return_distributions(historical, mean, std, bins=90,
                              title="Daily SPY log returns against the GBM assumption"):
    """The historical daily returns beside the normal GBM assumes they follow.

    The left panel shows both over the whole range, where they agree. The right
    panel is the left tail on a log frequency scale, which is where they do not
    -- and where a protective put earns its premium.
    """
    # Returns arrive as decimal fractions; both axes read in percent, and the
    # densities are rescaled to match so they still integrate to one.
    historical = np.asarray(historical, dtype=float) * 100.0
    mean, std = mean * 100.0, std * 100.0
    grid = np.linspace(historical.min() * 1.05, historical.max() * 1.05, 600)
    normal = (np.exp(-0.5 * ((grid - mean) / std) ** 2)
              / (std * np.sqrt(2 * np.pi)))

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.5))

    for ax in (left, right):
        ax.hist(historical, bins=bins, density=True, histtype="stepfilled",
                color=EMPIRICAL_COLOUR, alpha=0.30, linewidth=0)
        ax.hist(historical, bins=bins, density=True, histtype="step",
                color=EMPIRICAL_COLOUR, linewidth=2.0,
                label=f"observed ({historical.size} days)")
        ax.plot(grid, normal, color=GBM_COLOUR, linewidth=2.0,
                label="normal assumed by GBM")
        ax.xaxis.set_major_formatter(percent(1))

    _finish(left, "The whole distribution", "daily log return", "density")

    cutoff = float(np.percentile(historical, 5))
    left_edge = historical.min() * 1.08
    right.set_xlim(left_edge, cutoff)
    right.set_yscale("log")
    # Empty histogram bins would otherwise drag the log axis to 1e-19 and
    # squash the comparison into the top centimetre. The floor is set just
    # under the normal density at the far edge, which is the quantity the
    # observed bars are supposed to tower over.
    floor = float(np.exp(-0.5 * ((left_edge - mean) / std) ** 2)
                  / (std * np.sqrt(2 * np.pi)))
    right.set_ylim(bottom=max(floor / 3.0, 1e-9))
    right.axvline(cutoff, color=REFERENCE, linestyle="--", linewidth=1.2)
    right.annotate("worst 5% of days", xy=(cutoff, right.get_ylim()[1]),
                   xytext=(-6, -14), textcoords="offset points", fontsize=9,
                   color=REFERENCE, ha="right")
    _finish(right, "The left tail, log scale", "daily log return", "density")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig, (left, right)


def plot_quantile_comparison(probabilities, empirical, simulated,
                             horizon_days=10,
                             title="Ten-day outcomes: bootstrap against GBM"):
    """Matched quantiles of the two horizon distributions.

    Points on the diagonal mean the two engines agree at that probability.
    Departures below the line on the left say the bootstrap puts more weight in
    the loss tail than GBM does.
    """
    empirical = np.asarray(empirical, dtype=float) * 100.0
    simulated = np.asarray(simulated, dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=(8.5, 7))
    lo = float(min(empirical.min(), simulated.min()))
    hi = float(max(empirical.max(), simulated.max()))
    pad = 0.05 * (hi - lo)
    line = np.array([lo - pad, hi + pad])

    ax.plot(line, line, color=REFERENCE, linestyle="--", linewidth=1.4,
            label="the two agree", zorder=1)
    ax.scatter(simulated, empirical, s=34, color=EMPIRICAL_COLOUR,
               edgecolor="white", linewidth=0.8, zorder=3,
               label="matched quantiles")

    for probability, mark in ((0.01, "1st percentile"), (0.05, "5th")):
        index = int(np.argmin(np.abs(np.asarray(probabilities) - probability)))
        ax.annotate(mark, xy=(simulated[index], empirical[index]),
                    xytext=(10, -4), textcoords="offset points", fontsize=9,
                    color=REFERENCE)

    ax.xaxis.set_major_formatter(percent(0))
    ax.yaxis.set_major_formatter(percent(0))
    ax.set_aspect("equal", adjustable="box")
    return fig, _finish(ax, title,
                        f"GBM {horizon_days}-day log return",
                        f"bootstrap {horizon_days}-day log return")


def plot_risk_model_comparison(portfolios, gbm_values, bootstrap_values,
                               measure="99% CVaR", as_percent=False,
                               title=None):
    """One group per portfolio, one bar per risk model."""
    x = np.arange(len(portfolios), dtype=float)
    width, gap = 0.36, 0.02

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bars_gbm = ax.bar(x - width / 2 - gap, gbm_values, width,
                      color=GBM_COLOUR, label="GBM Monte Carlo")
    bars_boot = ax.bar(x + width / 2 + gap, bootstrap_values, width,
                       color=EMPIRICAL_COLOUR, label="historical bootstrap")

    formatter = (lambda v: f"{v:.2f}%") if as_percent else (lambda v: f"${v:,.0f}")
    for bars in (bars_gbm, bars_boot):
        for bar in bars:
            ax.annotate(formatter(bar.get_height()),
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, color=REFERENCE)

    for i, (a, b) in enumerate(zip(gbm_values, bootstrap_values)):
        if a:
            ax.annotate(f"{(b - a) / a * 100:+.1f}%", xy=(x[i], max(a, b)),
                        xytext=(0, 20), textcoords="offset points",
                        ha="center", fontsize=10, fontweight="bold",
                        color=REFERENCE)

    ax.set_xticks(x)
    ax.set_xticklabels(portfolios)
    ax.set_ylim(top=max(max(gbm_values), max(bootstrap_values)) * 1.24)
    ax.yaxis.set_major_formatter(percent(2) if as_percent else dollars(0))
    unit = "percent of portfolio value" if as_percent else "loss [$]"
    return fig, _finish(ax, title or f"{measure} under two risk models",
                        "portfolio", f"{measure} [{unit}]")


def plot_stress_test(shocks, stock_loss_percent, protected_loss_percent,
                     put_values, spot_prices=None,
                     title="What a crash does to each portfolio"):
    """Loss against the size of the fall, with the put's value underneath.

    Two panels rather than two y-axes on one: the loss and the option value are
    different quantities on different scales, and overlaying them would invent
    a relationship the data does not contain.
    """
    shocks = np.asarray(shocks, dtype=float) * 100.0
    order = np.argsort(shocks)
    shocks = shocks[order]
    stock_loss_percent = np.asarray(stock_loss_percent, dtype=float)[order]
    protected_loss_percent = np.asarray(protected_loss_percent, dtype=float)[order]
    put_values = np.asarray(put_values, dtype=float)[order]

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9.5, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.0]})

    top.axhline(0.0, color=REFERENCE, linewidth=1.2)
    top.plot(shocks, stock_loss_percent, color=LSMC, marker="o", markersize=8,
             linewidth=2.2, label="SPY only")
    top.plot(shocks, protected_loss_percent, color=BENCHMARK, marker="s",
             markersize=8, linewidth=2.2, linestyle="--", label="SPY + put")
    top.fill_between(shocks, protected_loss_percent, stock_loss_percent,
                     where=stock_loss_percent >= protected_loss_percent,
                     color=BENCHMARK, alpha=0.12, linewidth=0,
                     label="loss avoided by the hedge")

    for x, a, b in zip(shocks, stock_loss_percent, protected_loss_percent):
        if a - b > 0.4:
            top.annotate(f"{a - b:.1f} pts", xy=(x, (a + b) / 2),
                         xytext=(8, 0), textcoords="offset points",
                         fontsize=9, color=REFERENCE, va="center")

    top.yaxis.set_major_formatter(percent(0))
    _finish(top, title, "", "loss, measured from each portfolio's own start")

    bottom.plot(shocks, put_values, color=BENCHMARK, marker="s", markersize=8,
                linewidth=2.2, label="put value per share at the horizon")
    for x, v in zip(shocks, put_values):
        bottom.annotate(f"${v:,.2f}", xy=(x, v), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=9,
                        color=REFERENCE)
    bottom.set_ylim(bottom=0.0, top=put_values.max() * 1.22)
    bottom.xaxis.set_major_formatter(percent(0))
    bottom.yaxis.set_major_formatter(dollars(0))
    _finish(bottom, "", "shock to SPY over the ten-day horizon",
            "put value [$ per share]")

    fig.tight_layout()
    return fig, (top, bottom)


def plot_loss_distributions_by_model(gbm_losses, bootstrap_losses,
                                     gbm_var=None, bootstrap_var=None,
                                     bins=140, label="SPY only",
                                     title="Ten-day loss distribution under each risk model"):
    """The two engines' loss distributions for one portfolio, overlaid."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    edges = np.histogram_bin_edges(
        np.concatenate([gbm_losses, bootstrap_losses]), bins)

    for losses, colour, name in ((gbm_losses, GBM_COLOUR, "GBM Monte Carlo"),
                                 (bootstrap_losses, EMPIRICAL_COLOUR,
                                  "historical bootstrap")):
        ax.hist(losses, bins=edges, histtype="stepfilled", color=colour,
                alpha=0.28, linewidth=0)
        ax.hist(losses, bins=edges, histtype="step", color=colour,
                linewidth=2.0, label=name)

    for var, colour, name in ((gbm_var, GBM_COLOUR, "GBM"),
                              (bootstrap_var, EMPIRICAL_COLOUR, "bootstrap")):
        if var is not None:
            ax.axvline(var, color=colour, linestyle="--", linewidth=1.6)
            ax.annotate(f"99% VaR\n{name}\n${var:,.0f}",
                        xy=(var, ax.get_ylim()[1]), xytext=(5, -46),
                        textcoords="offset points", color=colour, fontsize=9)

    ax.axvline(0.0, color=REFERENCE, linewidth=1.0)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(dollars(0))
    return fig, _finish(ax, f"{title}  ({label})",
                        "loss over 10 trading days [$]",
                        "number of scenarios (log scale)")
