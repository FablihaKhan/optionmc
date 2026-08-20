"""Executive overview: the whole project in about thirty seconds.

Deliberately no large tables. This page is the argument, not the evidence --
what the problem is, what the engine computed, what it did to the tail risk,
and which contract came out of the optimizer. The detail lives one click away
on the pages that own it.

Every number is read from the cached pipeline results. Nothing on this page is
written into the source; if a phase has not been run, its section says which
command produces it and the rest of the page still renders.
"""
import numpy as np
import streamlit as st

import config
from ui import components as c
from ui import data_loader as dl
from ui import explanations
from ui import interactive_charts as ic
from ui import state
from ui.formatters import (count, money, percent, price, strike_label,
                           volatility)

CATEGORY_LABELS = {
    "cheapest": ("Cheapest hedge", "lowest premium"),
    "strongest": ("Strongest protection", "largest 99% CVaR reduction"),
    "most_efficient": ("Best efficiency", "most CVaR avoided per $1"),
    "balanced": ("Balanced", "equal weight on protection and cost"),
}


def _section(title, note=None):
    st.markdown(f"#### {title}")
    if note:
        st.caption(note)


def render():
    snapshot = dl.load_snapshot()
    c.page_header(
        "Overview",
        "From Monte Carlo option pricing to real portfolio protection.")
    c.timestamp_badge(snapshot, state.get("data_mode"), state.LIVE)

    if snapshot is None:
        c.warning_card(
            "No market snapshot on disk",
            "Run <code>python experiments/fetch_market_data.py</code>, or "
            "<code>python main.py --skip-fetch</code> to build everything from "
            "the cached data.", kind="bad")
        return

    explain = state.explain_simply()

    # --- 1. the problem -----------------------------------------------------
    _section("The problem, in three steps")
    left, middle, right = st.columns(3, gap="small")
    with left:
        c.metric_card("1 · Price", "What is an American put worth?",
                      "Early exercise makes it worth more than the European "
                      "price, and there is no closed form for it.", accent=True)
    with middle:
        c.metric_card("2 · Protect", "How much downside does it remove?",
                      "Measured as VaR and CVaR over ten trading days, from a "
                      "simulated distribution rather than a rule of thumb.",
                      accent=True)
    with right:
        c.metric_card("3 · Decide", "Which put is worth its cost?",
                      "Protection and premium rise together, so the answer is "
                      "a trade-off and not a single best contract.",
                      accent=True)

    # --- 2. market snapshot -------------------------------------------------
    _section("Market snapshot",
             "The frozen inputs every number below is conditional on.")
    c.metric_row([
        (f"{snapshot.ticker} spot", money(snapshot.spot),
         f"close of {snapshot.as_of}"),
        ("Selected put", strike_label(snapshot.strike),
         f"{snapshot.strike / snapshot.spot:.1%} of spot"),
        ("Days to expiry", count(snapshot.days_to_expiry),
         f"expires {snapshot.expiry}"),
        ("Market put price", money(snapshot.market_put_price),
         f"{snapshot.price_source} of {money(snapshot.bid)} / "
         f"{money(snapshot.ask)}"),
    ])

    # --- 3. the numerical engine --------------------------------------------
    pricing = dl.load_table("baseline_pricing")
    row = pricing.iloc[0] if pricing is not None and len(pricing) else None
    if row is not None:
        _section("Numerical engine",
                 "Two independent methods, and the early exercise that "
                 "separates this from the base project.")
        c.metric_row([
            ("LSMC American put", price(row["lsmc_price"]),
             f"standard error {price(row['lsmc_std_error'])}"),
            ("CRR binomial benchmark", price(row["binomial_price"]),
             f"{config.BINOMIAL_N_STEPS:,} steps, no shared code"),
            ("Relative difference",
             percent(row["relative_error_vs_binomial"] * 100),
             "simulation against lattice"),
            ("Early exercise premium", price(row["early_exercise_premium"]),
             f"{row['early_exercise_fraction']:.1%} of paths exercise early"),
        ])
        if explain:
            c.what_does_this_mean(explanations.pricing(
                lsmc=float(row["lsmc_price"]),
                binomial=float(row["binomial_price"]),
                market=float(row["market_price"]),
                std_error=float(row["lsmc_std_error"]),
                early_premium=float(row["early_exercise_premium"]),
                exercise_fraction=float(row["early_exercise_fraction"])))

    # --- 4. portfolio protection --------------------------------------------
    risk = dl.load_table("risk_dollars")
    risk_percent = dl.load_table("risk_percent")
    headline = None
    if risk is not None and len(risk) >= 3:
        unhedged, protected = risk.iloc[0], risk.iloc[1]
        _section(
            "Portfolio protection",
            f"{config.SHARES} SPY shares against the same shares plus one put, "
            f"over {config.RISK_HORIZON_DAYS} trading days and "
            f"{config.N_RISK_SCENARIOS:,} scenarios.")
        c.metric_row([
            ("99% VaR, unhedged", money(unhedged["var_99"]),
             "where the worst 1% begins"),
            ("99% VaR, hedged", money(protected["var_99"]),
             "the same threshold, with the put"),
            ("99% CVaR, unhedged", money(unhedged["cvar_99"]),
             "average loss inside that worst 1%"),
            ("99% CVaR, hedged", money(protected["cvar_99"]),
             "average loss inside that worst 1%"),
        ])

        headline = (float(risk_percent.iloc[2]["cvar_99"])
                    if risk_percent is not None and len(risk_percent) >= 3
                    else float(risk.iloc[2]["cvar_99"]))
        st.write("")
        _, centre, _ = st.columns([1, 2, 1])
        with centre:
            c.hero_metric(
                "Tail-risk reduction", percent(headline),
                "99% CVaR, each portfolio measured against its own value")
        if explain:
            c.what_does_this_mean(explanations.risk_measures(
                var=float(unhedged["var_99"]), cvar=float(unhedged["cvar_99"]),
                level=0.99,
                portfolio_value=float(unhedged["initial_value"])))

    # --- 5. optimizer recommendations ---------------------------------------
    rankings = dl.load_table("hedge_rankings")
    if rankings is not None and len(rankings):
        _section("Which put would you buy?",
                 "Four categories, four different questions. None of them is "
                 "labelled best.")
        indexed = rankings.set_index("category")
        columns = st.columns(4, gap="small")
        for column, key in zip(columns, CATEGORY_LABELS):
            if key not in indexed.index:
                continue
            entry = indexed.loc[key]
            title, subtitle = CATEGORY_LABELS[key]
            with column:
                c.recommendation_card(
                    title, strike_label(entry["strike"]),
                    [("Premium", money(entry["premium_cost"])),
                     ("Cost", percent(entry["hedge_cost_percent"])),
                     ("99% CVaR cut", percent(entry["cvar_99_reduction"]))],
                    note=subtitle)
        if explain:
            best = indexed.loc["most_efficient"]
            c.what_does_this_mean(explanations.hedging(
                cvar_reduction=float(best["cvar_99_reduction"]),
                cost_percent=float(best["hedge_cost_percent"]),
                saved_per_dollar=float(
                    best["cvar_99_saved_per_premium_dollar"])))

    # --- 6. one visual story ------------------------------------------------
    losses = dl.load_losses("portfolio")
    grid = dl.load_pricing_grid()
    if losses is not None and grid is not None and risk is not None:
        _section("The argument in one row",
                 "The risk, the instrument that changes it, and the result. "
                 "Panels 1 and 3 share an axis, so the shortened tail is "
                 "visible rather than asserted.")
        st.plotly_chart(
            ic.loss_story(
                losses["losses_unhedged"], losses["losses_protected"],
                grid["spots"], grid["prices"], grid["strike"], snapshot.spot,
                var_unhedged=float(risk.iloc[0]["var_99"]),
                var_protected=float(risk.iloc[1]["var_99"])),
            width="stretch", config={"displayModeBar": False})

    # --- 7. model robustness ------------------------------------------------
    models = dl.load_table("risk_models")
    diagnostics = dl.load_table("return_diagnostics")
    gbm_cvar = bootstrap_cvar = None
    if models is not None and len(models):
        unhedged_rows = models[models["put_cost_basis"] == "unhedged"]
        published = unhedged_rows[unhedged_rows["risk_model"] == "GBM Monte Carlo"]
        empirical = unhedged_rows[
            unhedged_rows["risk_model"] == "historical bootstrap"]
        if len(published) and len(empirical):
            gbm_cvar = float(published["cvar_99_dollars"].iloc[0])
            bootstrap_cvar = float(empirical["cvar_99_dollars"].iloc[0])
            harsher = ("historical bootstrap" if bootstrap_cvar > gbm_cvar
                       else "GBM")
            _section(
                "Does the answer depend on the risk model?",
                "The same portfolio, priced by the same grid, with scenarios "
                "from a normal model and from resampled history.")
            hedged = models[(models["put_cost_basis"] != "unhedged")
                            & (models["portfolio"].str.contains(
                                str(int(snapshot.strike))))]
            gbm_cut = float(hedged[hedged["risk_model"] == "GBM Monte Carlo"]
                            ["cvar_99_reduction"].iloc[0]) if len(hedged) else None
            boot_cut = float(hedged[hedged["risk_model"] == "historical bootstrap"]
                             ["cvar_99_reduction"].iloc[0]) if len(hedged) else None
            c.metric_row([
                ("GBM 99% CVaR", money(gbm_cvar), "normal daily returns"),
                ("Bootstrap 99% CVaR", money(bootstrap_cvar),
                 "resampled from observed days"),
                ("More conservative here", harsher,
                 percent(abs(bootstrap_cvar - gbm_cvar)
                         / min(bootstrap_cvar, gbm_cvar) * 100) + " apart"),
                ("Hedge still works",
                 f"{percent(gbm_cut)} / {percent(boot_cut)}"
                 if gbm_cut is not None else "—",
                 "99% CVaR cut under GBM / bootstrap"),
            ])
            if explain:
                kurtosis = None
                if diagnostics is not None and len(diagnostics):
                    observed = diagnostics[
                        diagnostics["label"] == "observed daily"]
                    if len(observed):
                        kurtosis = float(observed["excess_kurtosis"].iloc[0])
                c.what_does_this_mean(explanations.risk_models(
                    gbm_cvar=gbm_cvar, bootstrap_cvar=bootstrap_cvar,
                    excess_kurtosis=kurtosis))

    # --- 8. the project in one sentence -------------------------------------
    metrics = dl.load_table("cross_section_metrics")
    heldout_mae = n_heldout = None
    if metrics is not None and len(metrics):
        crr = metrics[(metrics["model"] == "crr")
                      & (metrics["scope"] == "overall")]
        if len(crr):
            heldout_mae = float(crr["mae"].iloc[0])
            n_heldout = int(crr["n"].iloc[0])

    candidates = dl.load_table("hedge_candidates")
    sentence = explanations.project_summary(
        early_premium=float(row["early_exercise_premium"]) if row is not None else None,
        relative_error=(float(row["relative_error_vs_binomial"]) * 100
                        if row is not None else None),
        heldout_mae=heldout_mae, n_heldout=n_heldout,
        cvar_reduction=headline,
        n_candidates=len(candidates) if candidates is not None else None,
        bootstrap_gap=(abs(bootstrap_cvar - gbm_cvar)
                       / min(bootstrap_cvar, gbm_cvar) * 100
                       if gbm_cvar and bootstrap_cvar else None))

    _section("The project in one sentence")
    c.callout(sentence, kind="good")

    missing = dl.missing_items()
    if len(missing):
        st.caption(f"{len(missing)} result files are not on disk, so some "
                   f"sections above are hidden. Run "
                   f"`python main.py --skip-fetch` to build them.")

    with st.expander("Colour key used across every page and figure"):
        c.colour_key()
