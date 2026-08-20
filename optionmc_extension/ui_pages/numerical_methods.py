"""Inside the numerical engine: the methods, their evidence, and the invariants.

This is the page that makes the case that the project is numerical analysis
rather than a finance dashboard with a chart library. Every claim on it is a
measurement from the experiment tables, and the invariants at the bottom are
re-derived from those tables each time the page loads rather than read from a
recorded verdict.
"""
import numpy as np
import pandas as pd
import streamlit as st

import config
from ui import components as c
from ui import compute
from ui import data_loader as dl
from ui import explanations
from ui import interactive_charts as ic
from ui import state
from ui.formatters import count, money, percent, price

PIPELINE = [
    ("European Monte Carlo",
     "The base OptionMC package: simulate terminal prices, average the "
     "discounted payoff. Exact for a European option and useless for an "
     "American one."),
    ("Full GBM paths",
     "Terminal prices are no longer enough. An exercise decision needs the "
     "whole trajectory, so the simulation keeps every step."),
    ("Longstaff-Schwartz regression",
     "At each step, regress the realised discounted continuation value on a "
     "polynomial in the current price, using in-the-money paths only."),
    ("American early exercise",
     "Compare immediate payoff against the fitted continuation value, stop "
     "where exercising wins, and discount what remains."),
    ("CRR lattice validation",
     "An independent method sharing no code. Agreement between the two is the "
     "main evidence that either is right."),
    ("Pricing grid and interpolation",
     "Price once at 65 spots, read back by PCHIP for every scenario. Pricing "
     "each scenario directly would be a nested Monte Carlo."),
    ("Real-world risk simulation",
     "Fifty thousand horizon scenarios under the historical drift, or "
     "resampled from observed returns. Never the pricing measure."),
    ("VaR and CVaR",
     "Empirical quantile and tail mean of the loss distribution, with "
     "bootstrap intervals because a tail statistic from a finite sample is "
     "itself uncertain."),
    ("Hedge optimisation",
     "Score real listed contracts on protection against cost, and identify "
     "which of them nothing else dominates."),
]

METHODS = [
    ("Monte Carlo simulation",
     "Estimates an expectation by averaging over sampled paths.",
     "Pricing every option, and generating the risk scenarios.",
     "The only tractable route once the payoff depends on a whole path."),
    ("Least-squares regression",
     "Fits a polynomial to noisy data by minimising squared residuals.",
     "Estimating the continuation value at each exercise date.",
     "The continuation value is a conditional expectation, and this is how "
     "Longstaff-Schwartz estimates one from simulated paths."),
    ("Backward induction",
     "Solves a problem by starting at the end and working towards the "
     "present.",
     "Both the LSMC recursion and the CRR tree.",
     "An optimal stopping problem cannot be solved forwards: today's decision "
     "depends on what the option will be worth later."),
    ("Binomial approximation",
     "Replaces continuous price movement with a recombining lattice.",
     "The CRR benchmark, and inverting market prices for implied volatility.",
     "Deterministic, so it carries no sampling error and can act as the "
     "reference the simulation is judged against."),
    ("Root finding",
     "Solves f(x) = 0 by bracketing and bisection, here Brent's method.",
     "Implied volatility: which sigma reproduces this market price?",
     "The map from volatility to price has no inverse in closed form."),
    ("PCHIP interpolation",
     "A shape-preserving cubic through every data point.",
     "Reading the pricing grid at arbitrary spots, and fitting the "
     "volatility smile.",
     "It passes through the data without the overshoot a natural spline can "
     "produce between points -- a negative option price would be unusable."),
    ("Empirical quantiles",
     "Reads a percentile straight off the sorted sample.",
     "VaR, and the tail average behind CVaR.",
     "It assumes no distribution, which is the point when the whole question "
     "is whether the distribution is normal."),
    ("Bootstrap resampling",
     "Rebuilds a distribution by drawing observations with replacement.",
     "The historical risk engine, and the confidence intervals on VaR "
     "and CVaR.",
     "It carries the data's own shape -- skew and fat tails included -- "
     "instead of a fitted approximation of it."),
]


def render():
    snapshot = dl.load_snapshot()
    c.page_header("Inside the numerical engine",
                  "The methods underneath every number in this dashboard, and "
                  "the evidence that each one works.")
    c.timestamp_badge(snapshot, state.get("data_mode"), state.LIVE)

    _evolution()
    _methods()
    _convergence()
    _discretisation()
    _regression()
    _interpolation()
    _invariants()


def _evolution():
    st.markdown("#### From the base paper to a decision")
    st.caption("Each step exists because the one before it could not answer "
               "the next question.")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        c.pipeline(PIPELINE[:5])
    with right:
        c.pipeline(PIPELINE[5:])


def _methods():
    st.markdown("#### The methods, and why each one is here")
    for start in range(0, len(METHODS), 4):
        for column, entry in zip(st.columns(4, gap="small"),
                                 METHODS[start:start + 4]):
            name, what, where, why = entry
            with column:
                c.metric_card(
                    name, what,
                    f"<b>Where</b>: {where}<br><b>Why</b>: {why}", accent=True)
        st.write("")


def _convergence():
    frame = dl.load_table("convergence")
    if frame is None:
        return
    from src.replication import fit_convergence_order

    st.markdown("#### Convergence: does more work buy accuracy?")
    fit = fit_convergence_order(frame["n_paths"].to_numpy(),
                                frame["rmse"].to_numpy())
    st.plotly_chart(
        ic.convergence_with_order(
            frame["n_paths"], frame["mean_price"],
            frame["mean_price"] - 2 * frame["std_price"],
            frame["mean_price"] + 2 * frame["std_price"],
            float(frame["benchmark"].iloc[0]), frame["rmse"],
            fitted_order=fit["order"]),
        width="stretch", config={"displayModeBar": False})

    c.metric_row([
        ("Fitted order", f"{fit['order']:.3f}",
         "slope of log error against log paths"),
        ("Theoretical order", "-0.500",
         "the Monte Carlo rate, error ∝ 1/√N"),
        ("Replications per point", count(frame["n_replications"].iloc[0]),
         "independent seeds, so the order is not one seed's luck"),
        ("Largest run", count(frame["n_paths"].max()),
         f"RMSE {money(frame['rmse'].min(), 4)}"),
    ])
    st.caption(
        f"Quadrupling the paths should halve the error, and a fitted order of "
        f"{fit['order']:.3f} against the theoretical −0.5 is what that looks "
        f"like. This is the check that matters more than any single price: an "
        f"estimator converging at the right rate is behaving, while one that "
        f"is merely close at one sample size may be close by accident.")

    if state.explain_simply():
        c.what_does_this_mean(explanations.numerical_methods(
            convergence_order=fit["order"]))


def _discretisation():
    frame = dl.load_table("discretization")
    if frame is None:
        return

    st.markdown("#### Discretisation: how many exercise dates are enough?")
    st.plotly_chart(
        ic.discretisation_chart(
            frame["n_steps"], frame["mean_price"],
            frame["mc_error_vs_bermudan"], frame["discretisation_error"],
            frame["mean_runtime_sec"],
            float(frame["bermudan_benchmark"].iloc[0])),
        width="stretch", config={"displayModeBar": False})
    st.caption(
        "The two errors are separated on purpose. Too few exercise dates makes "
        "the option formally Bermudan and under-prices it, always in the same "
        "direction; Monte Carlo noise is symmetric. Added together they can "
        "cancel, and a coarse grid can look accurate for the wrong reason — "
        "which is why the experiment prices a Bermudan tree on the same "
        "exercise dates to isolate each part.")

    st.dataframe(
        frame[["n_steps", "mean_price", "mc_error_vs_bermudan",
               "discretisation_error", "total_error_vs_american",
               "mean_runtime_sec"]],
        hide_index=True, width="stretch",
        column_config={
            "n_steps": st.column_config.NumberColumn("Exercise dates"),
            "mean_price": st.column_config.NumberColumn("Mean price",
                                                        format="%.4f"),
            "mc_error_vs_bermudan": st.column_config.NumberColumn(
                "Monte Carlo error", format="%.4f",
                help="Against a Bermudan tree on the same exercise dates, so "
                     "discretisation is held fixed."),
            "discretisation_error": st.column_config.NumberColumn(
                "Discretisation error", format="%.4f",
                help="Bermudan against fully American: the cost of exercising "
                     "on a finite grid."),
            "total_error_vs_american": st.column_config.NumberColumn(
                "Total", format="%.4f"),
            "mean_runtime_sec": st.column_config.NumberColumn(
                "Runtime", format="%.3f s"),
        })


def _regression():
    frame = dl.load_table("regression")
    if frame is None:
        return

    st.markdown("#### Regression basis: is a higher degree better?")
    degrees = sorted(frame["degree"].unique())
    path_counts = sorted(frame["n_paths"].unique())
    series = {
        str(frame[frame["degree"] == degree]["basis"].iloc[0]):
            frame[frame["degree"] == degree].sort_values("n_paths")
            ["mean_price"].to_numpy()
        for degree in degrees}

    st.plotly_chart(
        ic.regression_degree_chart(path_counts, series,
                                   float(frame["benchmark"].iloc[0])),
        width="stretch", config={"displayModeBar": False})

    largest = frame[frame["n_paths"] == max(path_counts)]
    st.caption(
        "Every degree converges to the same place, and at the largest sample "
        "they differ by less than the standard error. The answer to \"does a "
        "higher polynomial help?\" is measured rather than assumed — and the "
        "measurement says the basis matters far less than the number of paths.")

    st.dataframe(
        largest[["degree", "basis", "mean_price", "std_error_of_mean", "bias",
                 "rmse", "mean_runtime_sec"]].sort_values("degree"),
        hide_index=True, width="stretch",
        column_config={
            "degree": st.column_config.NumberColumn("Degree"),
            "basis": st.column_config.TextColumn("Basis"),
            "mean_price": st.column_config.NumberColumn("Mean price",
                                                        format="%.4f"),
            "std_error_of_mean": st.column_config.NumberColumn(
                "Standard error", format="%.4f"),
            "bias": st.column_config.NumberColumn(
                "Bias", format="%.4f",
                help="Against the Bermudan benchmark on the same exercise "
                     "dates."),
            "rmse": st.column_config.NumberColumn("RMSE", format="%.4f"),
            "mean_runtime_sec": st.column_config.NumberColumn(
                "Runtime", format="%.3f s"),
        })
    c.callout(
        "More complexity is not automatically better. A higher degree fits the "
        "continuation value more flexibly, and with few paths it also fits the "
        "sampling noise — which produces an exercise rule that looks better "
        "than it is and a price biased upwards. The grid in this project uses "
        f"degree {config.GRID_DEGREE} because that was measured to be the "
        f"turning point, not because higher sounded safer.")


def _interpolation():
    grid = dl.load_table("pricing_grid")
    accuracy = dl.load_table("interpolation_accuracy")
    if grid is None:
        return

    st.markdown("#### Interpolation: pricing once and reading it back")
    st.plotly_chart(
        ic.grid_against_lattice(
            grid["spot"], grid["american_put_price"], grid["binomial_price"],
            grid["intrinsic_value"],
            float(grid["spot"].iloc[0] / grid["moneyness"].iloc[0])),
        width="stretch", config={"displayModeBar": False})
    st.caption(
        f"{len(grid)} nodes priced once, then read back by interpolation for "
        f"each of {config.N_RISK_SCENARIOS:,} risk scenarios. Pricing every "
        f"scenario directly would be a nested Monte Carlo — fifty thousand "
        f"simulations inside a simulation — and would take days for a number "
        f"that is available in seconds this way.")

    if accuracy is not None:
        isolated = accuracy[accuracy["stage"] == "a_interpolation_only"]
        total = accuracy[accuracy["stage"] == "c_total_vs_binomial"]
        st.markdown("**How much the interpolation itself costs**")
        st.dataframe(
            isolated[["method", "max_absolute_error", "mean_absolute_error",
                      "bias"]],
            hide_index=True, width="stretch",
            column_config={
                "method": st.column_config.TextColumn("Method"),
                "max_absolute_error": st.column_config.NumberColumn(
                    "Worst error", format="%.5f"),
                "mean_absolute_error": st.column_config.NumberColumn(
                    "Mean error", format="%.5f"),
                "bias": st.column_config.NumberColumn("Bias", format="%.5f"),
            })
        chosen = isolated[isolated["method"] == config.INTERPOLATION_METHOD]
        if len(chosen) and len(total):
            against_tree = total[total["method"] == config.INTERPOLATION_METHOD]
            st.caption(
                f"Measured against directly priced points that reuse the same "
                f"random numbers, so what is left is interpolation error and "
                f"nothing else: worst "
                f"{price(float(chosen['max_absolute_error'].iloc[0]), 5)} per "
                f"share. Against the lattice the gap is "
                f"{price(float(against_tree['max_absolute_error'].iloc[0]), 5)}"
                f", and the difference between those two numbers is the LSMC's "
                f"own bias at the nodes, not the interpolant's.")

        if state.explain_simply():
            worst = float(isolated["max_absolute_error"].max())
            c.what_does_this_mean(explanations.numerical_methods(
                interpolation_error=worst, grid_nodes=len(grid)))


def _invariants():
    st.divider()
    checks = compute.sanity_checks()
    c.check_list(checks, title="Invariants that must hold")
    st.caption(
        "These are properties no correct implementation can violate, so a red "
        "row would mean a bug rather than an interesting finding. Each one "
        "names the file it was checked against.")
