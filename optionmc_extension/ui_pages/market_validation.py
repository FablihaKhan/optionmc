"""Out-of-sample validation: does the framework generalise, or only reprice?

There is a way of testing an option pricer that looks rigorous and proves
nothing: take a market price, solve for the volatility that reproduces it, put
that volatility back into the model, and report a tiny error. The model has
been handed the answer.

This page shows the version that does not do that. Within each expiry the
usable strikes were split by position -- even calibrates, odd is held out --
and only the calibration contracts had their implied volatilities solved. Each
held-out contract was priced from a curve fitted through its *neighbours*, and
only then compared with its own quote.

Everything here is read from the saved validation run, so the page is instant
and needs no network.
"""
import numpy as np
import pandas as pd
import streamlit as st

import config
from src.cross_section_validation import MIN_PRICE_FOR_PERCENTAGE
from ui import components as c
from ui import compute
from ui import data_loader as dl
from ui import explanations
from ui import interactive_charts as ic
from ui import state
from ui.formatters import count, money, percent, volatility

FAIRNESS = """
The obvious test of an option pricer is circular. Take a contract's market
price, solve for the volatility that reproduces it, feed that volatility back
into the model, and of course the model returns the price you started from. The
error is tiny and it measures nothing but arithmetic.

**What was done instead.** For each expiry the usable strikes were sorted and
split by position: even index calibrates, odd index is held out. Only the
calibration contracts had their American implied volatilities solved, against
the CRR tree. A PCHIP curve was fitted through *those points alone*. Each
held-out contract then read its volatility off that curve — interpolated from
the strikes either side of it — and was priced with both CRR and LSMC. Its own
market price was read afterwards, only to score the prediction.

**Why that is fairer.** A held-out contract's quote never touches anything used
to predict it. The first and last strike of each expiry are forced into the
calibration set, so every held-out point is interpolated between two fitted
points rather than extrapolated past the end of the curve.

**What it does and does not show.** It shows the American implied-volatility
surface is smooth enough that a strike left out of the fit can be repriced from
its neighbours to within a few cents, and that two independent pricers agree to
within the simulation's own sampling error. It does **not** show the model
beats the market: the pricer valuing the held-out contract is the same one that
inverted the calibration quotes, so what is tested is the interpolation and the
pricing pipeline, not a view about where prices should be.
"""


def render():
    snapshot = dl.load_snapshot()
    c.page_header("Does the model generalise?",
                  "Out-of-sample testing across real SPY options.")
    c.timestamp_badge(snapshot, state.get("data_mode"), state.LIVE)

    universe = dl.load_table("cross_section")
    calibration = dl.load_table("calibration_set")
    heldout = dl.load_table("heldout")
    metrics = dl.load_table("cross_section_metrics")

    if any(frame is None for frame in (universe, calibration, heldout, metrics)):
        c.warning_card(
            "The cross-section validation has not been run",
            "Run <code>python experiments/cross_section_validation_experiment"
            ".py</code>. Add <code>--refresh</code> during market hours to "
            "cover three maturities instead of one.", kind="bad")
        with st.expander("Why is this a fairer validation?", expanded=True):
            st.markdown(FAIRNESS)
        return

    ok = heldout[heldout["prediction_status"] == "ok"]
    crr = metrics[(metrics["model"] == "crr") & (metrics["scope"] == "overall")]
    lsmc = metrics[(metrics["model"] == "lsmc") & (metrics["scope"] == "overall")]

    _headline(universe, calibration, ok, crr, lsmc)
    _split(universe, calibration, ok, snapshot)
    _smile(calibration, ok)
    _market_against_model(ok, crr, lsmc)
    _errors(ok, metrics)
    _density()
    _footer(ok, crr, calibration)


def _headline(universe, calibration, ok, crr, lsmc):
    expiries = sorted(universe["expiry"].unique())
    st.caption(
        f"{len(expiries)} expiry" + ("" if len(expiries) == 1 else "/expiries")
        + " · " + ", ".join(expiries)
        + f" · strikes {config.CROSS_SECTION_MIN_MONEYNESS:.0%}"
          f"-{config.CROSS_SECTION_MAX_MONEYNESS:.0%} of spot")

    c.metric_row([
        ("Contracts in the band", count(len(universe)),
         "before any quote filter"),
        ("Passing the filters",
         count(len(calibration) + len(ok)),
         f"thinned to ${config.CROSS_SECTION_STRIKE_SPACING:g} strike spacing"),
        ("Used to fit the curve", count(len(calibration)),
         "their implied volatilities were solved"),
        ("Held out", count(len(ok)),
         "priced from their neighbours only"),
    ])
    st.write("")

    if len(crr) and len(lsmc):
        crr_row, lsmc_row = crr.iloc[0], lsmc.iloc[0]
        c.metric_row([
            ("CRR mean absolute error", money(crr_row["mae"], 4),
             f"against a mean quote of {money(float(ok['mid'].mean()))}"),
            ("CRR RMSE", money(crr_row["rmse"], 4),
             "larger than the MAE only if a few contracts miss badly"),
            ("Median error", percent(crr_row["median_abs_pct_error"]),
             f"over the {int(crr_row['n_for_percentage'])} priced above "
             f"{money(MIN_PRICE_FOR_PERCENTAGE)}"),
            ("LSMC mean absolute error", money(lsmc_row["mae"], 4),
             "its own sampling noise is the larger part of this"),
        ])


def _split(universe, calibration, ok, snapshot):
    st.markdown("#### Which contracts were used for what")
    st.caption("The two sets never overlap. That is the whole design, so it is "
               "drawn rather than asserted.")

    used = pd.concat([
        calibration.assign(role="calibration")[["strike", "expiry", "role"]],
        ok.assign(role="test")[["strike", "expiry", "role"]],
    ], ignore_index=True).sort_values("strike")

    spot = float(snapshot.spot) if snapshot is not None else float(
        universe["spot"].iloc[0])
    st.plotly_chart(
        ic.calibration_split_chart(used["strike"], used["role"],
                                   used["expiry"], spot),
        width="stretch", config={"displayModeBar": False})

    shared = set(zip(calibration["expiry"], calibration["strike"])) & \
        set(zip(ok["expiry"], ok["strike"]))
    c.callout(
        f"{len(calibration)} calibration contracts, {len(ok)} held out, "
        f"<b>{len(shared)} in both</b>. Every held-out strike sits between two "
        f"calibration strikes, so none of them is extrapolated past the end of "
        f"the curve.", kind="good" if not shared else "bad")


def _smile(calibration, ok):
    st.markdown("#### The volatility curve, fitted to calibration quotes only")
    expiry = str(calibration["expiry"].iloc[0])
    curve = compute.smile_curve(expiry)
    if curve is None:
        c.warning_card("Not enough calibration points to fit a curve", "")
        return

    st.plotly_chart(
        ic.smile_chart(curve["calibration_x"], curve["calibration_y"],
                       curve["x"], curve["y"],
                       ok["log_moneyness"], ok["interpolated_vol"]),
        width="stretch", config={"displayModeBar": False})
    st.caption(
        f"The skew runs from {volatility(curve['y'].max())} on the low strikes "
        f"to {volatility(curve['y'].min())} above the money — priced from real "
        f"quotes, not assumed. The held-out diamonds show the volatility each "
        f"contract was *given*; their own implied volatilities are absent on "
        f"purpose, because drawing them here would suggest they shaped the "
        f"curve.")


def _market_against_model(ok, crr, lsmc):
    st.markdown("#### Predicted price against the quote")
    hover = [f"K={row.strike:g}, {row.moneyness:.1%} of spot, {row.expiry}"
             for row in ok.itertuples()]
    st.plotly_chart(
        ic.scatter_with_identity(
            ok["mid"].tolist(),
            {"CRR binomial": ok["crr_price"].tolist(),
             "LSMC": ok["lsmc_price"].tolist()},
            xlabel="market mid [$]", ylabel="model price [$]",
            hover_text=hover),
        width="stretch", config={"displayModeBar": False})

    if len(lsmc):
        within = int((ok["lsmc_error"].abs()
                      <= 2 * ok["lsmc_std_error"]).sum())
        st.caption(
            f"At this scale every point sits on the diagonal, which is the "
            f"finding — the residual panel below is where the accuracy shows. "
            f"{within} of {len(ok)} LSMC errors fall inside that contract's own "
            f"two-standard-error range, so the simulation's disagreement with "
            f"the market is mostly its own sampling noise.")


def _errors(ok, metrics):
    st.markdown("#### Where the error sits")
    strike_tab, money_tab, dte_tab, heat_tab, raw_tab = st.tabs(
        ["Error vs strike", "Error vs moneyness", "Error vs expiry",
         "Error heatmap", "Raw held-out data"])

    errors = {"CRR binomial": ok["crr_error"], "LSMC": ok["lsmc_error"]}
    hover = [f"K={row.strike:g}, market ${row.mid:,.3f}"
             for row in ok.itertuples()]

    with strike_tab:
        st.plotly_chart(
            ic.error_chart(ok["strike"], errors, "strike [$]",
                           band=ok["lsmc_std_error"], hover_text=hover,
                           x_prefix="$"),
            width="stretch", config={"displayModeBar": False})
        st.caption("The CRR error grows gently with the strike because the "
                   "options themselves get more expensive; as a share of price "
                   "it is flat.")

    with money_tab:
        st.plotly_chart(
            ic.error_chart(ok["moneyness"] * 100, errors,
                           "strike as a percentage of spot",
                           band=ok["lsmc_std_error"], hover_text=hover),
            width="stretch", config={"displayModeBar": False})

    with dte_tab:
        if ok["expiry"].nunique() > 1:
            st.plotly_chart(
                ic.error_chart(ok["days_to_expiry"], errors,
                               "days to expiry", band=ok["lsmc_std_error"],
                               hover_text=hover),
                width="stretch", config={"displayModeBar": False})
        else:
            c.callout(
                "Only one maturity is quoted in the cached chain, so there is "
                "nothing to plot against. Run "
                "<code>python experiments/cross_section_validation_experiment"
                ".py --refresh</code> during trading hours to fetch three "
                "expiries; the chart appears automatically. It is not drawn "
                "from stale last-trade prices, which are recorded up to an "
                "hour apart and would show up as curve roughness the model "
                "would be blamed for.")

    with heat_tab:
        buckets = pd.cut(ok["moneyness"], [0.90, 0.95, 1.00, 1.05],
                         labels=["90-95%", "95-100%", "100-105%"])
        pivot = (ok.assign(bucket=buckets)
                   .pivot_table(index="expiry", columns="bucket",
                                values="crr_abs_error", aggfunc="mean",
                                observed=False))
        st.plotly_chart(ic.error_heatmap(pivot), width="stretch",
                        config={"displayModeBar": False})
        st.caption("Absolute error rises towards the money because the options "
                   "there are worth more; the relative error does not.")

    with raw_tab:
        columns = ["strike", "moneyness", "mid", "interpolated_vol",
                   "crr_price", "crr_error", "lsmc_price", "lsmc_error",
                   "lsmc_std_error"]
        st.dataframe(
            ok[columns].sort_values("strike"), hide_index=True,
            width="stretch",
            column_config={
                "strike": st.column_config.NumberColumn("Strike", format="%.0f"),
                "moneyness": st.column_config.NumberColumn(
                    "Moneyness", format="percent"),
                "mid": st.column_config.NumberColumn(
                    "Market mid", format="dollar",
                    help="Read only after the prediction was made."),
                "interpolated_vol": st.column_config.NumberColumn(
                    "Volatility used", format="percent",
                    help="Interpolated from neighbouring calibration strikes."),
                "crr_price": st.column_config.NumberColumn(
                    "CRR", format="%.4f"),
                "crr_error": st.column_config.NumberColumn(
                    "CRR error", format="%.4f"),
                "lsmc_price": st.column_config.NumberColumn(
                    "LSMC", format="%.4f"),
                "lsmc_error": st.column_config.NumberColumn(
                    "LSMC error", format="%.4f"),
                "lsmc_std_error": st.column_config.NumberColumn(
                    "LSMC s.e.", format="%.4f",
                    help="The simulation's own sampling error on that price."),
            })

        summary = metrics[["model", "scope", "n", "mae", "rmse", "bias",
                           "median_abs_pct_error", "max_abs_error"]]
        st.markdown("**Accuracy, overall and by maturity group**")
        st.dataframe(summary, hide_index=True, width="stretch")


def _density():
    spacing = dl.load_table("spacing_study")
    if spacing is None or len(spacing) < 2:
        return

    st.markdown("#### How far apart can the calibration strikes be?")
    st.caption("The held-out set is held fixed throughout; only the "
               "calibration grid is thinned, so this isolates the curve's own "
               "interpolation error.")
    st.plotly_chart(
        ic.spacing_study_chart(
            spacing["spacing"],
            {"CRR binomial": spacing["crr_mae"], "LSMC": spacing["lsmc_mae"]},
            noise_floor=float(spacing["lsmc_std_error_mean"].mean())),
        width="stretch", config={"displayModeBar": False})

    best = spacing.loc[spacing["crr_mae"].idxmin()]
    finest, coarsest = spacing.iloc[0], spacing.iloc[-1]
    c.callout(
        f"The error does not fall as the grid gets finer. It is lowest at "
        f"<b>${best['spacing']:g}</b> spacing ({money(best['crr_mae'], 4)}), "
        f"worse at ${coarsest['spacing']:g} ({money(coarsest['crr_mae'], 4)}) "
        f"where the curve has too far to reach, and also worse at "
        f"${finest['spacing']:g} ({money(finest['crr_mae'], 4)}) where PCHIP "
        f"threads the penny-wide quantisation of the quotes and carries that "
        f"wobble into the held-out strikes. Densest is not best.")


def _footer(ok, crr, calibration):
    st.divider()
    with st.expander("Why is this a fairer validation?", expanded=True):
        st.markdown(FAIRNESS)

    if state.explain_simply() and len(crr):
        st.markdown("#### What does this mean?")
        row = crr.iloc[0]
        for line in explanations.validation(
                mae=float(row["mae"]), rmse=float(row["rmse"]),
                n_calibration=len(calibration), n_heldout=len(ok),
                median_pct=float(row["median_abs_pct_error"]),
                mean_quote=float(ok["mid"].mean())):
            st.markdown(f"- {line}")

    c.assumption_box([
        ("Split", "Deterministic and fixed in advance: sorted strikes, even "
                  "index calibrates, odd index is held out. Nobody can tune it "
                  "after seeing the errors."),
        ("Implied volatility", "Solved against the CRR American tree. "
                               "Inverting Black-Scholes on an American quote "
                               "would absorb the early-exercise premium into "
                               "sigma."),
        ("Interpolation", "PCHIP over log-moneyness, which passes through "
                          "every point without the overshoot a cubic spline "
                          "can produce between them."),
        ("Percentages", "Reported as a median and only over contracts priced "
                        "above ten cents, so one cheap option cannot distort "
                        "the figure."),
        ("Scope", "One expiry has live quotes in the cached chain, so this "
                  "validates across strikes. The maturity dimension needs a "
                  "fetch during trading hours."),
    ])
