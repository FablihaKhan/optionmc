"""Tail risk under two models, and crash scenarios that carry no probability.

Value-at-Risk and Expected Shortfall answer "how bad does it get at a stated
probability". A stress test answers a blunter question that does not depend on
the model believing anything: if the market falls twenty percent in ten days,
what happens?

Both are here because they fail differently. A risk model can understate the
tail; a stress test cannot, because it does not model the tail at all -- but it
also cannot tell you how likely the scenario is.

Everything is read from the saved runs, so the page is instant and offline.
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
from ui.formatters import (count, money, percent, price, signed_money,
                           volatility)

VAR_AGAINST_CVAR = """
**Value at Risk** answers *where does the bad tail begin?* A 99% VaR of $5,598
means that on the worst 1% of ten-day outcomes the loss is at least that large.
It is a threshold, and it says nothing whatever about what lies beyond it.

**Conditional Value at Risk**, also called Expected Shortfall, answers the
harder question: *once we are already in that worst 1%, how bad is it on
average?* It is the mean of everything past the VaR threshold, so it is always
the larger of the two, and it is the one that notices how far the tail actually
reaches.

The difference matters here. Two distributions can share a VaR to the dollar
and have completely different CVaRs — one stops just past the threshold, the
other keeps going. A protective put works precisely in that region, so a model
reported only through VaR would undervalue it.

This is also why the Basel market-risk framework moved from VaR to Expected
Shortfall for tail capital: a threshold can be gamed by a position that hides
its losses just past it, and an average over the tail cannot.
"""


def _levels():
    return list(config.CONFIDENCE_LEVELS)


def render():
    snapshot = dl.load_snapshot()
    c.page_header("How bad can the loss get?",
                  "Tail risk under two models, and crash scenarios that carry "
                  "no probability at all.")
    c.timestamp_badge(snapshot, state.get("data_mode"), state.LIVE)

    risk = dl.load_table("risk_dollars")
    models = dl.load_table("risk_models")
    stress = dl.load_table("stress")
    losses = dl.load_losses("portfolio")
    bootstrap = dl.load_losses("bootstrap")

    if risk is None or losses is None:
        c.warning_card(
            "The portfolio risk run has not been done",
            "Run <code>python experiments/portfolio_risk.py</code>, or "
            "<code>python main.py --skip-fetch</code> for everything.",
            kind="bad")
        return

    st.caption(f"{config.N_RISK_SCENARIOS:,} scenarios over "
               f"{config.RISK_HORIZON_DAYS} trading days · "
               f"{config.SHARES} SPY shares against the same shares plus one "
               f"put")

    gbm_tab, boot_tab, stress_tab, compare_tab = st.tabs(
        ["GBM Monte Carlo", "Historical bootstrap", "Stress test",
         "Compare models"])

    with gbm_tab:
        _gbm(risk, losses)
    with boot_tab:
        _bootstrap(models, bootstrap)
    with stress_tab:
        _stress(stress)
    with compare_tab:
        _compare(models)


def _measure_cards(row, prefix="", basis=None):
    items = []
    for level in _levels():
        key = f"{level:.0%}".replace("%", "")
        for stat, name in (("var", "VaR"), ("cvar", "CVaR")):
            column = f"{stat}_{key}{prefix}"
            items.append((f"{level:.0%} {name}", money(row[column]),
                          "where the bad tail begins" if stat == "var"
                          else "average loss inside that tail"))
    c.metric_row(items)
    if basis:
        st.caption(basis)


def _gbm(risk, losses):
    st.markdown("#### Unhedged: 100 SPY shares")
    _measure_cards(risk.iloc[0])
    st.write("")
    st.markdown("#### Protected: the same shares plus one put")
    _measure_cards(risk.iloc[1])

    st.markdown("#### The two loss distributions")
    st.plotly_chart(
        ic.loss_distribution(
            losses["losses_unhedged"], losses["losses_protected"],
            var_lines=[
                ("95% VaR, unhedged", float(risk.iloc[0]["var_95"]),
                 ic.LEVEL_RAMP[0]),
                ("99% VaR, unhedged", float(risk.iloc[0]["var_99"]),
                 ic.LEVEL_RAMP[1]),
            ]),
        width="stretch", config={"displayModeBar": False})
    st.caption(
        f"The counts are on a log axis. The worst 1% is five hundred scenarios "
        f"out of {config.N_RISK_SCENARIOS:,}, and on a linear axis that region "
        f"— the whole reason for buying a put — would be a flat line at the "
        f"bottom of the chart.")

    if state.explain_simply():
        c.what_does_this_mean(explanations.risk_measures(
            var=float(risk.iloc[0]["var_99"]),
            cvar=float(risk.iloc[0]["cvar_99"]), level=0.99,
            portfolio_value=float(risk.iloc[0]["initial_value"])))


def _bootstrap(models, bootstrap):
    if models is None or bootstrap is None:
        c.warning_card(
            "The bootstrap engine has not been run",
            "Run <code>python experiments/bootstrap_risk_experiment.py</code>.",
            kind="warn")
        return

    rows = models[models["risk_model"] == "historical bootstrap"]
    unhedged = rows[rows["put_cost_basis"] == "unhedged"]
    hedged = rows[rows["put_cost_basis"] != "unhedged"]

    st.markdown("#### Unhedged, with scenarios resampled from observed days")
    _measure_cards(unhedged.iloc[0], prefix="_dollars")
    if len(hedged):
        st.write("")
        st.markdown(f"#### Protected ({hedged.iloc[0]['portfolio']})")
        _measure_cards(hedged.iloc[0], prefix="_dollars")

    st.markdown("#### The two loss distributions")
    st.plotly_chart(
        ic.loss_distribution(
            bootstrap["bootstrap_unhedged"], bootstrap["bootstrap_protected"],
            var_lines=[
                ("99% VaR, unhedged",
                 float(unhedged.iloc[0]["var_99_dollars"]), ic.LEVEL_RAMP[1]),
            ]),
        width="stretch", config={"displayModeBar": False})

    st.markdown("#### The returns these scenarios are drawn from")
    history = compute.historical_returns()
    if history is None:
        return
    summary = history["summary"]
    st.plotly_chart(
        ic.return_distribution(history["returns"], summary["mean"],
                               summary["std"]),
        width="stretch", config={"displayModeBar": False})
    c.metric_row([
        ("Observed days", count(summary["n"]),
         f"{history['start']} to {history['end']}"),
        ("Worst single day", percent(summary["min"] * 100),
         "the bootstrap can draw this ten times over"),
        ("Excess kurtosis", f"{summary['excess_kurtosis']:+.2f}",
         "a normal distribution has zero"),
        ("Skewness", f"{summary['skewness']:+.2f}",
         "negative means the left tail is longer"),
    ])
    st.caption(
        "Nothing is fitted here. Each scenario draws ten of these observed "
        "days with replacement and sums them, so whatever shape the history "
        "has is carried straight into the horizon. One limitation is worth "
        "naming: drawing days independently discards volatility clustering, so "
        "a real crash — several bad days in a row — is less likely under this "
        "bootstrap than in the market.")


def _stress(stress):
    if stress is None:
        c.warning_card(
            "The stress test has not been run",
            "Run <code>python experiments/stress_test_experiment.py</code>.",
            kind="warn")
        return

    portfolios = list(dict.fromkeys(stress["portfolio"]))
    chosen = st.selectbox("Hedge", portfolios, key="stress_portfolio")
    table = stress[stress["portfolio"] == chosen].sort_values("shock")

    st.plotly_chart(
        ic.stress_chart(table["shock"], table["stock_only_loss_percent"],
                        table["protected_loss_percent"],
                        table["put_value_per_share"]),
        width="stretch", config={"displayModeBar": False})

    deepest = table.iloc[0]
    flat = table[np.isclose(table["shock"], 0.0)]
    c.callout(
        f"The protected loss flattens near "
        f"<b>{percent(table['protected_loss_percent'].max())}</b> however deep "
        f"the crash goes, while the unhedged loss tracks the shock one for one "
        f"to {percent(deepest['stock_only_loss_percent'])}. The put does not "
        f"remove the loss — it caps how fast it grows once the market is "
        f"through the strike."
        + (f" In a flat market it costs "
           f"{money(abs(float(flat['hedge_benefit_dollars'].iloc[0])))} of time "
           f"value, which is the price of protection that turned out to be "
           f"unnecessary." if len(flat) else ""))

    st.markdown("#### Every scenario")
    columns = ["shock", "spy_price", "stock_only_loss", "protected_loss",
               "put_value_per_share", "hedge_benefit_dollars",
               "stock_only_loss_percent", "protected_loss_percent"]
    st.dataframe(
        table[columns], hide_index=True, width="stretch",
        column_config={
            "shock": st.column_config.NumberColumn("Shock", format="percent"),
            "spy_price": st.column_config.NumberColumn("SPY", format="dollar"),
            "stock_only_loss": st.column_config.NumberColumn(
                "Stock loss", format="dollar"),
            "protected_loss": st.column_config.NumberColumn(
                "Protected loss", format="dollar",
                help="Measured from the protected portfolio's own starting "
                     "value, which includes what the put cost."),
            "put_value_per_share": st.column_config.NumberColumn(
                "Put value", format="dollar", help="Per share, at the horizon, "
                                                   "with the correct remaining "
                                                   "maturity."),
            "hedge_benefit_dollars": st.column_config.NumberColumn(
                "Hedge benefit", format="dollar",
                help="Stock loss minus protected loss. Negative when the "
                     "market does not fall."),
            "stock_only_loss_percent": st.column_config.NumberColumn(
                "Stock loss %", format="%.2f%%"),
            "protected_loss_percent": st.column_config.NumberColumn(
                "Protected loss %", format="%.2f%%"),
        })

    agreement = dl.load_table("stress_pricers")
    if agreement is not None:
        block = agreement[agreement["hedge"] == chosen]
        if len(block):
            worst = float(block["difference"].abs().max())
            st.caption(
                f"The horizon put value was computed twice — once through the "
                f"interpolated pricing grid and once through a "
                f"{config.BINOMIAL_N_STEPS:,}-step tree. Largest disagreement "
                f"{price(worst)} per share, "
                f"{money(worst * config.CONTRACT_MULTIPLIER)} per contract, "
                f"against hedge benefits in the thousands.")

    if state.explain_simply():
        c.what_does_this_mean(explanations.stress(
            worst_unhedged=float(table["stock_only_loss_percent"].max()),
            worst_protected=float(table["protected_loss_percent"].max()),
            flat_cost=(float(flat["hedge_benefit_dollars"].iloc[0])
                       if len(flat) else None),
            largest_benefit=float(table["hedge_benefit_dollars"].max()),
            largest_shock=float(
                table.loc[table["hedge_benefit_dollars"].idxmax(), "shock"])))


def _compare(models):
    if models is None:
        c.warning_card(
            "The model comparison has not been run",
            "Run <code>python experiments/bootstrap_risk_experiment.py</code>.",
            kind="warn")
        return

    unhedged = models[models["put_cost_basis"] == "unhedged"]
    engines = list(dict.fromkeys(models["risk_model"]))

    published = unhedged[unhedged["risk_model"] == "GBM Monte Carlo"]
    empirical = unhedged[unhedged["risk_model"] == "historical bootstrap"]
    gbm_cvar = float(published["cvar_99_dollars"].iloc[0])
    boot_cvar = float(empirical["cvar_99_dollars"].iloc[0])
    harsher = ("historical bootstrap" if boot_cvar > gbm_cvar
               else "GBM Monte Carlo")

    c.callout(
        f"The more conservative 99% CVaR in this snapshot is produced by the "
        f"<b>{harsher}</b> — {money(max(gbm_cvar, boot_cvar))} against "
        f"{money(min(gbm_cvar, boot_cvar))}, "
        f"{percent(abs(boot_cvar - gbm_cvar) / min(boot_cvar, gbm_cvar) * 100)} "
        f"apart. Which model wins is read off the calculation, not decided in "
        f"advance.", kind="info")

    st.markdown("#### The unhedged position under each engine")
    st.plotly_chart(
        ic.grouped_bars(
            [f"{level:.0%} {stat}" for level in _levels()
             for stat in ("VaR", "CVaR")],
            {engine: [float(unhedged[unhedged["risk_model"] == engine]
                            [f"{stat.lower()}_{level:.0%}".replace("%", "")
                             + "_dollars"].iloc[0])
                      for level in _levels() for stat in ("VaR", "CVaR")]
             for engine in engines},
            xlabel="measure", ylabel="loss [$]", value_format="${:,.0f}"),
        width="stretch", config={"displayModeBar": False})

    st.markdown("#### Side by side")
    columns = ["risk_model", "portfolio", "var_95_dollars", "cvar_95_dollars",
               "var_99_dollars", "cvar_99_dollars", "cvar_99_reduction"]
    st.dataframe(
        models[columns], hide_index=True, width="stretch",
        column_config={
            "risk_model": st.column_config.TextColumn("Risk model"),
            "portfolio": st.column_config.TextColumn("Portfolio"),
            "var_95_dollars": st.column_config.NumberColumn(
                "95% VaR", format="dollar"),
            "cvar_95_dollars": st.column_config.NumberColumn(
                "95% CVaR", format="dollar"),
            "var_99_dollars": st.column_config.NumberColumn(
                "99% VaR", format="dollar"),
            "cvar_99_dollars": st.column_config.NumberColumn(
                "99% CVaR", format="dollar"),
            "cvar_99_reduction": st.column_config.NumberColumn(
                "99% CVaR cut", format="%.2f%%",
                help="Against the unhedged position under the same engine."),
        })

    quantiles = dl.load_table("horizon_quantiles")
    if quantiles is not None:
        st.markdown("#### Where the two engines disagree")
        st.plotly_chart(
            ic.quantile_chart(
                quantiles["empirical"], quantiles["simulated"],
                xlabel=f"GBM {config.RISK_HORIZON_DAYS}-day log return",
                ylabel=f"bootstrap {config.RISK_HORIZON_DAYS}-day log return"),
            width="stretch", config={"displayModeBar": False})
        st.caption(
            "Points on the diagonal mean the two engines agree at that "
            "probability. They track each other through the middle and part "
            "company in the loss tail on the lower left, which is the only "
            "region either model is being asked about.")

    diagnostics = dl.load_table("return_diagnostics")
    if diagnostics is not None:
        with st.expander("Why they differ: the first two moments match"):
            st.markdown(
                "The bootstrap and the drift-matched GBM are built to share a "
                "mean and a standard deviation, so any gap in VaR or CVaR is "
                "the *shape* of the distribution and nothing else.")
            st.dataframe(
                diagnostics[["label", "n", "mean", "std", "skewness",
                             "excess_kurtosis", "p01", "min"]],
                hide_index=True, width="stretch")
            st.caption(
                "The published GBM arm sits slightly below the drift-matched "
                "one because src/gbm.py subtracts sigma-squared over two from "
                "the drift it is given, while the estimator supplies a mean "
                "log return. Small, conservative for VaR, and left unchanged "
                "so every earlier result still reproduces.")

    st.divider()
    st.markdown("#### VaR against CVaR")
    st.markdown(VAR_AGAINST_CVAR)

    if state.explain_simply():
        history = compute.historical_returns()
        c.what_does_this_mean(explanations.risk_models(
            gbm_cvar=gbm_cvar, bootstrap_cvar=boot_cvar,
            excess_kurtosis=(history["summary"]["excess_kurtosis"]
                             if history else None)))

    c.assumption_box([
        ("Measure", "Both engines simulate under the real-world measure. "
                    "Option values at the horizon stay risk-neutral and come "
                    "from the cached pricing grid."),
        ("GBM", "Normal daily log returns with constant volatility, "
                "calibrated to the same history the bootstrap resamples."),
        ("Bootstrap", "No distribution is fitted. Ten observed daily returns "
                      "drawn with replacement and summed. Volatility "
                      "clustering is discarded by that independence."),
        ("Stress", "Deterministic shocks with no probability attached, priced "
                   "at the correct remaining maturity."),
    ])
