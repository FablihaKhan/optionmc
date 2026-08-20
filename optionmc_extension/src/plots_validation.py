"""Figures for the out-of-sample cross-section validation.

Separate from `plots.py` and `plots_hedge.py` so neither of those changes, but
importing the same palette so a colour means the same thing everywhere: blue is
the LSMC estimate, orange the CRR benchmark, aqua the observed market, grey a
reference that is not a series.

One rule is enforced here rather than left to the caller: the held-out
contracts' own implied volatilities are never drawn on the smile. Showing them
next to the fitted curve would suggest they helped shape it, which is the
precise thing this validation is built to avoid.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, NullLocator

from .plots import (BENCHMARK, LSMC, MARKET, REFERENCE, _finish, dollars,
                    percent)


def plot_market_vs_model(market, crr, lsmc, lsmc_std_error=None,
                         title="Held-out contracts: model against market"):
    """Predicted price against quoted price, with the residuals beside it.

    The scatter alone would be uninformative here: at this error scale every
    point sits on the diagonal. The right panel is where the accuracy actually
    shows, so both are drawn and the residual panel carries the LSMC's own
    Monte Carlo band -- the noise floor a simulation cannot get below.
    """
    market = np.asarray(market, dtype=float)
    crr = np.asarray(crr, dtype=float)
    lsmc = np.asarray(lsmc, dtype=float)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.5))

    lo = float(np.nanmin([market.min(), crr.min(), lsmc.min()]))
    hi = float(np.nanmax([market.max(), crr.max(), lsmc.max()]))
    pad = 0.05 * (hi - lo)
    line = np.array([lo - pad, hi + pad])
    left.plot(line, line, color=REFERENCE, linestyle="--", linewidth=1.4,
              label="perfect prediction (y = x)", zorder=1)
    left.scatter(market, crr, s=80, color=BENCHMARK, edgecolor="white",
                 linewidth=1.5, label="CRR binomial", zorder=3)
    left.scatter(market, lsmc, s=80, color=LSMC, marker="D", edgecolor="white",
                 linewidth=1.5, label="LSMC", zorder=2)
    left.xaxis.set_major_formatter(dollars(0))
    left.yaxis.set_major_formatter(dollars(0))
    _finish(left, "Predicted against quoted", "market mid [$]",
            "model price [$]")

    if lsmc_std_error is not None:
        band = 2.0 * np.asarray(lsmc_std_error, dtype=float)
        order = np.argsort(market)
        right.fill_between(market[order], -band[order], band[order],
                           color=LSMC, alpha=0.15, linewidth=0,
                           label="LSMC +/- 2 standard errors")
    right.axhline(0.0, color=REFERENCE, linewidth=1.4, linestyle="--")
    right.scatter(market, crr - market, s=80, color=BENCHMARK,
                  edgecolor="white", linewidth=1.5, label="CRR binomial",
                  zorder=3)
    right.scatter(market, lsmc - market, s=80, color=LSMC, marker="D",
                  edgecolor="white", linewidth=1.5, label="LSMC", zorder=2)
    right.xaxis.set_major_formatter(dollars(0))
    right.yaxis.set_major_formatter(dollars(2))
    _finish(right, "Residuals", "market mid [$]", "model minus market [$]")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig, (left, right)


def plot_volatility_smile(calibration_x, calibration_y, curve_x, curve_y,
                          test_x, test_y=None, expiry_label="",
                          title="Implied volatility smile from calibration strikes only"):
    """The fitted smile, the points that made it, and where it was queried.

    `test_y` is the interpolated volatility each held-out contract received --
    a value read off the curve, not a market observation. The held-out
    contracts' own market implied volatilities are deliberately absent.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    # Volatilities arrive as decimal fractions; the axis reads in percent.
    calibration_y = np.asarray(calibration_y, dtype=float) * 100.0
    curve_y = np.asarray(curve_y, dtype=float) * 100.0

    ax.plot(curve_x, curve_y, color=LSMC, linewidth=2.0, zorder=2,
            label="PCHIP smile through calibration points")
    ax.scatter(calibration_x, calibration_y, s=95, color=LSMC,
               edgecolor="white", linewidth=1.8, zorder=4,
               label=f"calibration strikes ({len(calibration_x)})")

    if test_y is None:
        for x in test_x:
            ax.axvline(x, color=BENCHMARK, linestyle=":", linewidth=1.2,
                       alpha=0.7, zorder=1)
    else:
        ax.scatter(test_x, np.asarray(test_y, dtype=float) * 100.0, s=95,
                   marker="D", facecolor="white", edgecolor=BENCHMARK,
                   linewidth=2.0, zorder=5,
                   label=f"held-out strikes, volatility read off the curve "
                         f"({len(test_x)})")

    ax.axvline(0.0, color=REFERENCE, linewidth=1.0)
    ax.annotate("at the money", xy=(0.0, ax.get_ylim()[0]), xytext=(4, 6),
                textcoords="offset points", fontsize=9, color=REFERENCE)
    ax.yaxis.set_major_formatter(percent(1))
    suffix = f"  ({expiry_label})" if expiry_label else ""
    return fig, _finish(ax, title + suffix, "log-moneyness  log(K / S0)",
                        "American implied volatility")


def plot_error_against(x, crr_error, lsmc_error, lsmc_std_error=None,
                       xlabel="strike as a percentage of spot",
                       title="Held-out pricing error", as_percent_axis=True):
    """Prediction error against any contract characteristic."""
    x = np.asarray(x, dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    if lsmc_std_error is not None:
        band = 2.0 * np.asarray(lsmc_std_error, dtype=float)
        order = np.argsort(x)
        ax.fill_between(x[order], -band[order], band[order], color=LSMC,
                        alpha=0.15, linewidth=0,
                        label="LSMC +/- 2 standard errors")

    ax.axhline(0.0, color=REFERENCE, linewidth=1.4, linestyle="--")
    ax.plot(x, crr_error, color=BENCHMARK, marker="o", markersize=7,
            linewidth=2.0, label="CRR binomial")
    ax.plot(x, lsmc_error, color=LSMC, marker="D", markersize=7,
            linewidth=2.0, linestyle="-", label="LSMC")

    if as_percent_axis:
        ax.xaxis.set_major_formatter(percent(1))
    ax.yaxis.set_major_formatter(dollars(2))
    return fig, _finish(ax, title, xlabel, "model minus market [$]")


def plot_calibration_split(strikes, roles, expiries, spot,
                           title="Which contracts were used for what"):
    """Where the calibration and held-out strikes sit, expiry by expiry.

    The point of the figure is that the two sets never coincide, which is
    easier to trust when it is visible than when it is asserted.
    """
    strikes = np.asarray(strikes, dtype=float)
    roles = np.asarray(roles)
    expiries = np.asarray(expiries)

    labels = list(dict.fromkeys(expiries))
    fig, ax = plt.subplots(figsize=(10, 1.6 + 0.9 * len(labels)))

    for row, expiry in enumerate(labels):
        here = expiries == expiry
        cal = here & (roles == "calibration")
        test = here & (roles == "test")
        ax.scatter(strikes[cal], np.full(cal.sum(), row), s=110, color=LSMC,
                   edgecolor="white", linewidth=1.5, zorder=3,
                   label="calibration" if row == 0 else None)
        ax.scatter(strikes[test], np.full(test.sum(), row), s=110, marker="D",
                   facecolor="white", edgecolor=BENCHMARK, linewidth=2.0,
                   zorder=4, label="held out" if row == 0 else None)

    ax.axvline(spot, color=REFERENCE, linestyle="--", linewidth=1.4)
    ax.annotate(f"spot ${spot:,.2f}", xy=(spot, len(labels) - 0.5),
                xytext=(6, 0), textcoords="offset points", fontsize=9,
                color=REFERENCE, va="center")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.xaxis.set_major_formatter(dollars(0))
    ax.grid(axis="y", visible=False)
    return fig, _finish(ax, title, "strike [$]", "expiry")


def plot_spacing_study(spacings, crr_mae, lsmc_mae, lsmc_noise=None,
                       title="How far apart can the calibration strikes be?"):
    """Held-out accuracy as the calibration grid is thinned.

    The question a numerical-methods reader asks next: the smile interpolates
    well at this spacing, but where does it stop? Widening the gap between
    calibration points until the error grows answers it by measurement.
    """
    spacings = np.asarray(spacings, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(spacings, crr_mae, color=BENCHMARK, marker="o", markersize=7,
            linewidth=2.0, label="CRR binomial")
    ax.plot(spacings, lsmc_mae, color=LSMC, marker="D", markersize=7,
            linewidth=2.0, label="LSMC")
    if lsmc_noise is not None:
        ax.axhline(lsmc_noise, color=REFERENCE, linestyle=":", linewidth=1.6)
        ax.annotate("LSMC Monte Carlo noise floor",
                    xy=(spacings[0], lsmc_noise), xytext=(4, 5),
                    textcoords="offset points", fontsize=9, color=REFERENCE)

    # Log on x because the spacings span a factor of twenty; linear on y
    # because the errors span barely a factor of five and a log axis there
    # would only make a small difference look dramatic.
    ax.set_xscale("log")
    ax.set_xticks(spacings)
    ax.set_xticklabels([f"${s:g}" for s in spacings])
    # Matplotlib keeps drawing its own decade labels underneath the explicit
    # ticks, which collide with them.
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_formatter(dollars(3))
    return fig, _finish(ax, title, "spacing between calibration strikes",
                        "mean absolute error on held-out contracts [$]")


def plot_error_heatmap(pivot, title="Held-out error by expiry and moneyness"):
    """Error by expiry and moneyness bucket.

    A sequential single-hue ramp: the value is a magnitude, so the encoding is
    light to dark rather than a set of unrelated hues. Every cell also carries
    its number, so the figure does not rely on colour to be read.
    """
    rows, columns = pivot.shape
    fig, ax = plt.subplots(figsize=(1.7 * columns + 3.5, 0.9 * rows + 2.4))
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="Blues", aspect="auto")
    # Without this a single-expiry table stretches one row over the whole
    # figure and the cells stop reading as cells.
    ax.set_box_aspect(0.42 * rows)

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=0)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)

    threshold = np.nanmax(values) * 0.6 if np.isfinite(values).any() else 0.0
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                ax.text(j, i, "--", ha="center", va="center", fontsize=9,
                        color=REFERENCE)
                continue
            ax.text(j, i, f"${value:.3f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if value > threshold else "#1a1a1a")

    bar = fig.colorbar(image, ax=ax, shrink=0.85)
    bar.set_label("mean absolute error [$]")
    ax.grid(visible=False)
    return fig, _finish(ax, title, "moneyness bucket", "expiry", legend=False)
