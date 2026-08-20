"""Interactive pricing lab: run the method, do not just read its result.

This is the page where the numerical work is demonstrated. Everything on the
left is an input; nothing expensive happens until RUN PRICING is pressed. The
inputs live inside a form for exactly that reason -- without one, dragging a
slider would launch a fresh simulation on every pixel of movement.

The previous result stays on screen while a new one is computed, and is only
replaced when the new run finishes. A blank page mid-calculation looks like a
crash to anyone watching.
"""
import numpy as np
import streamlit as st

import config
from ui import components as c
from ui import compute
from ui import data_loader as dl
from ui import explanations
from ui import interactive_charts as ic
from ui import state
from ui.formatters import (count, money, percent, price, signed_money,
                           volatility)
from ui.state import PricingParams

RESULT_KEY = "pricing_lab_result"
PARAMS_KEY = "pricing_lab_params"

LSMC_EXPLANATION = """
**The problem.** An American put can be exercised on any day, so its value is
not an expectation over one payoff — it is the value of an optimal stopping
rule. There is no closed form for it.

**The idea (Longstaff and Schwartz, 2001).** At each step, a holder who is in
the money compares two numbers: what exercising pays *now*, and what continuing
is worth. The second is unknown, so it is estimated by regressing the
discounted cash flows that actually followed onto the current price.

**The recursion**, run backwards from expiry:

1. **Simulate** paths of the underlying under the risk-neutral drift `r − q`.
2. At the last step, the cash flow is the payoff `max(K − S_T, 0)`.
3. Step **backwards**. Discount the cash flow already carried by each path.
4. Keep only the paths that are **in the money** — the rest have no decision to
   make, and including them wastes the regression on a region where the answer
   is always "wait".
5. **Regress** the discounted future cash flow on a polynomial in the current
   price: `E[continuation | S] ≈ β₀ + β₁S + β₂S² (+ β₃S³)`.
6. **Exercise or wait**: where `K − S` exceeds the fitted continuation value,
   replace that path's cash flow with the immediate payoff and record the stop.
7. **Discount** everything back to today and average.

**Two details that matter.** The regressand is the cash flow that *realised*
along the path, not the fitted value — using the fit on both sides would let
the regression grade its own work. And the price is floored at the intrinsic
value, because an American option can always be exercised immediately.
"""


def input_form(snapshot):
    """Every pricing input, inside a form so nothing runs until submitted."""
    defaults = dict(
        spot=float(snapshot.spot), strike=float(snapshot.strike),
        maturity=float(snapshot.time_to_expiry),
        rate=float(snapshot.risk_free_rate),
        dividend=float(snapshot.dividend_yield),
        sigma=float(snapshot.historical_volatility))

    with st.form("pricing_lab_form", border=True):
        st.markdown("**Contract**")
        spot = st.number_input("Spot price", value=defaults["spot"],
                               min_value=1.0, step=1.0, format="%.2f")
        strike = st.number_input("Strike", value=defaults["strike"],
                                 min_value=1.0, step=1.0, format="%.2f")
        maturity = st.number_input(
            "Time to expiry (years)", value=defaults["maturity"],
            min_value=0.01, max_value=5.0, step=0.01, format="%.4f",
            help=f"The cached contract has {snapshot.days_to_expiry} days left.")

        st.markdown("**Market**")
        rate = st.number_input("Risk-free rate", value=defaults["rate"],
                               min_value=0.0, max_value=0.25, step=0.001,
                               format="%.4f")
        dividend = st.number_input("Dividend yield", value=defaults["dividend"],
                                   min_value=0.0, max_value=0.25, step=0.001,
                                   format="%.4f")
        sigma = st.number_input(
            "Volatility", value=defaults["sigma"], min_value=0.01,
            max_value=2.0, step=0.005, format="%.4f",
            help="Historical by default. The option's own implied volatility "
                 f"is {snapshot.implied_volatility:.4f}.")

        st.markdown("**Simulation**")
        n_paths = st.number_input(
            "LSMC paths", value=int(state.get("n_paths")), min_value=100,
            max_value=compute.MAX_PATHS, step=1_000,
            help=f"Capped at {compute.MAX_PATHS:,} so an interactive run stays "
                 "inside a few seconds.")
        n_steps = st.number_input("Exercise steps", value=int(state.get("n_steps")),
                                  min_value=5, max_value=250, step=5)
        degree = st.number_input("Polynomial degree",
                                 value=int(state.get("degree")),
                                 min_value=1, max_value=4, step=1)
        seed = st.number_input("Random seed", value=int(state.get("seed")),
                               min_value=0, max_value=100_000, step=1)

        run = st.form_submit_button("Run pricing", type="primary",
                                    width="stretch")

    return run, PricingParams(
        spot=float(spot), strike=float(strike), time_to_expiry=float(maturity),
        risk_free_rate=float(rate), dividend_yield=float(dividend),
        volatility=float(sigma), n_paths=int(n_paths), n_steps=int(n_steps),
        degree=int(degree), seed=int(seed))


def render():
    snapshot = dl.load_snapshot()
    c.page_header(
        "Pricing Lab",
        "Where the numerical method is demonstrated, not just its result.")
    c.timestamp_badge(snapshot, state.get("data_mode"), state.LIVE)

    if snapshot is None:
        c.warning_card(
            "No market snapshot on disk",
            "Run <code>python experiments/fetch_market_data.py</code> to give "
            "this page a contract to start from.", kind="bad")
        return

    controls, output = st.columns([1.05, 2.95], gap="medium")

    with controls:
        run, params = input_form(snapshot)

    if run:
        with output:
            with st.spinner("Simulating paths and running the backward "
                            "recursion…"):
                try:
                    result = compute.price_once(
                        params, binomial_steps=config.BINOMIAL_N_STEPS)
                except Exception as exc:              # noqa: BLE001
                    # The old result stays on screen; a bad input must not
                    # blank the page in front of an audience.
                    c.warning_card("That run could not be completed",
                                   "The previous result is still shown below.",
                                   kind="bad")
                    with st.expander("Technical detail"):
                        st.code(f"{type(exc).__name__}: {exc}")
                    result = None
            if result is not None:
                st.session_state[RESULT_KEY] = result
                st.session_state[PARAMS_KEY] = params

    result = st.session_state.get(RESULT_KEY)
    params = st.session_state.get(PARAMS_KEY)

    with output:
        if result is None:
            c.callout(
                "Set the inputs on the left and press <b>Run pricing</b>. "
                "Nothing is simulated until you do — this page never starts a "
                "calculation just because a number changed.")
            with st.expander("How Longstaff-Schwartz works", expanded=True):
                st.markdown(LSMC_EXPLANATION)
            return

        _results(params, result)

    _footer(params, result)


def _results(params, result):
    """Metrics and the four charts, once a run has completed."""
    c.metric_row([
        ("LSMC price", price(result["lsmc_price"]),
         f"{count(result['n_paths_used'])} paths, degree {params.degree}"),
        ("CRR price", price(result["binomial_american"]),
         f"{config.BINOMIAL_N_STEPS:,} steps"),
        ("Difference", signed_money(result["difference"], 4),
         percent(result["relative_difference"]) + " of the CRR price"),
        ("Standard error", price(result["lsmc_std_error"]),
         "of the LSMC estimate"),
    ])
    st.write("")
    c.metric_row([
        ("Early exercise premium", price(result["early_exercise_premium"]),
         "American minus European, same paths"),
        ("Paths exercising early",
         percent(result["early_exercise_fraction"] * 100),
         f"of {count(result['n_paths_used'])}"),
        ("European price", price(result["european_price"]),
         "terminal payoff only"),
        ("Intrinsic value", price(result["intrinsic"]),
         "the floor the price cannot fall below"),
    ])

    price_tab, exercise_tab, convergence_tab, raw_tab = st.tabs(
        ["Price", "Early exercise", "Convergence", "Raw results"])

    with price_tab:
        st.markdown("**Simulated paths under the risk-neutral drift r − q**")
        times, paths = compute.sample_paths(params)
        st.plotly_chart(ic.paths_chart(times, paths, strike=params.strike),
                        width="stretch", config={"displayModeBar": False})
        st.caption(
            "These are the paths the regression sees. The drift is r − q "
            "because this is pricing; the historical drift belongs to the risk "
            "pages and never appears here.")

        st.markdown("**LSMC against the lattice, across spot**")
        with st.spinner("Pricing across a range of spots…"):
            across = compute.price_across_spots(params)
        st.plotly_chart(ic.price_vs_spot(across, params.strike, params.spot),
                        width="stretch", config={"displayModeBar": False})
        worst = float(across["difference"].abs().max())
        st.caption(
            f"Two methods that share no code, over a {len(across)}-point "
            f"range. Largest disagreement {price(worst)} per share. The LSMC "
            f"curve reuses one set of paths rescaled to every spot, so its "
            f"shape is not a different random draw at each point.")

    with exercise_tab:
        st.markdown("**The estimated early-exercise boundary**")
        st.plotly_chart(
            ic.exercise_boundary_chart(result["time_remaining"],
                                       result["exercise_boundary"],
                                       params.strike),
            width="stretch", config={"displayModeBar": False})
        finite = np.isfinite(result["exercise_boundary"])
        st.caption(
            f"Below the line, exercising beats waiting. It rises towards the "
            f"strike as expiry approaches, because the time value that made "
            f"waiting worthwhile is running out. "
            f"{int(finite.sum())} of {finite.size} nodes had enough "
            f"in-the-money paths to fit a regression.")

        stops = result["stopping_step"][result["exercised_early"]]
        if stops.size:
            st.markdown("**When the paths stopped**")
            st.plotly_chart(
                ic.histogram_overlay(
                    {"exercised early": stops * params.time_to_expiry
                     / params.n_steps},
                    xlabel="time of exercise [years]", bins=params.n_steps),
                width="stretch", config={"displayModeBar": False})

    with convergence_tab:
        st.markdown("**Price against the number of paths**")
        with st.spinner("Running the sweep…"):
            sweep = compute.convergence_sweep(params)
        st.plotly_chart(ic.convergence_chart(sweep), width="stretch",
                        config={"displayModeBar": False})
        st.caption(
            "One replication per point — enough to show the trend, not enough "
            "to measure a convergence order. The thirty-seed study that fits "
            "the order lives on the Numerical Methods page.")
        st.dataframe(
            sweep[["n_paths", "price", "std_error", "absolute_error"]],
            width="stretch", hide_index=True)

    with raw_tab:
        st.markdown("**This run**")
        st.dataframe(compute.summary_frame(params, result), width="stretch",
                     hide_index=True)
        st.markdown("**Exercise boundary by node**")
        st.dataframe(compute.exercise_boundary_frame(result), width="stretch",
                     hide_index=True, height=280)


def _footer(params, result):
    """The reading of the result, and the method explained."""
    st.divider()
    relative = result["relative_difference"]
    if relative < 1.0:
        verdict = (
            f"The two independent numerical methods agree to "
            f"<b>{percent(relative)}</b> of the price"
            + (", inside the simulation's own two-standard-error range"
               if result["within_two_standard_errors"] else "")
            + ", which supports the implementation.")
        kind = "good"
    elif relative < 3.0:
        verdict = (
            f"The two methods differ by <b>{percent(relative)}</b>. That is "
            f"larger than the sampling noise usually explains — worth more "
            f"paths or more steps before trusting the number.")
        kind = "warn"
    else:
        verdict = (
            f"The two methods differ by <b>{percent(relative)}</b>. At this "
            f"distance neither should be treated as settled.")
        kind = "bad"

    st.markdown("#### What does this mean?")
    c.callout(verdict, kind=kind)

    if state.explain_simply():
        for line in explanations.pricing(
                lsmc=result["lsmc_price"], binomial=result["binomial_american"],
                std_error=result["lsmc_std_error"],
                early_premium=result["early_exercise_premium"],
                exercise_fraction=result["early_exercise_fraction"]):
            st.markdown(f"- {line}")

    with st.expander("How Longstaff-Schwartz works"):
        st.markdown(LSMC_EXPLANATION)

    c.assumption_box([
        ("Measure", "Pricing uses the risk-neutral drift r − q. The historical "
                    "drift never enters this page."),
        ("Dynamics", "Geometric Brownian motion with constant volatility, so "
                     "no volatility smile and no jumps."),
        ("Exercise", f"American exercise is approximated by {params.n_steps} "
                     f"equally spaced opportunities, which makes this formally "
                     f"a Bermudan option and a slight under-estimate."),
        ("Benchmark", f"The CRR tree at {config.BINOMIAL_N_STEPS:,} steps is "
                      f"treated as the reference, but it carries its own "
                      f"discretisation error."),
    ])
