"""The decision page: which real listed put is worth buying?

Everything here comes from contracts that were actually quoted. The strikes are
the nearest listed ones to 90, 92.5, 95, 97.5 and 100 percent of spot, the cost
is the ask an investor would pay, and each contract carries the volatility
implied by its own market price.

Two things this page refuses to do. It never names a single best put -- four
objectives answer four different questions, and which one matters is the
investor's choice. And it never draws a dominated hedge as if it were on the
frontier.

The expensive work -- fifty thousand scenarios per candidate -- runs only when
Analyze hedges is pressed. Changing the objective or the weights afterwards
just re-scores the table already in hand, so the recommendation moves
instantly.
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
from ui.formatters import (count, money, percent, price, strike_label,
                           volatility)

TABLE_KEY = "hedge_table"
OBJECTIVES = ("Cheapest", "Strongest protection", "Best efficiency",
              "Balanced")

HOW_TO_READ = """
- **Moving right** on the frontier means more expensive insurance.
- **Moving up** means more tail protection.
- The **upper-left** region is generally the attractive one: more protection
  for less money.
- Points on the blue line are **Pareto efficient** — nothing else available is
  both cheaper and stronger. A hollow point is beaten on both counts by
  something else on the board, and is never presented as a choice.
- **No hedge is universally optimal.** The cheapest one leaves the most risk;
  the strongest one costs the most; the most efficient one is usually neither.
  Which is right depends on what the investor is trying to do.
"""


def _sync_cost():
    st.session_state.hedge_cost_weight = round(
        1.0 - st.session_state.hedge_protection_weight, 2)


def _sync_protection():
    st.session_state.hedge_protection_weight = round(
        1.0 - st.session_state.hedge_cost_weight, 2)


def controls(snapshot):
    """The choices. None of them is expensive; only the button is."""
    st.session_state.setdefault("hedge_protection_weight", 0.5)
    st.session_state.setdefault("hedge_cost_weight", 0.5)

    top = st.columns([1.1, 0.9, 1.1, 1.2], gap="small")
    with top[0]:
        st.selectbox("Expiry", (snapshot.expiry,), key="hedge_expiry",
                     help="One expiry is quoted in the cached chain. The "
                          "cross-section page fetches more.")
    with top[1]:
        level = st.selectbox("Confidence level", config.CONFIDENCE_LEVELS,
                             index=len(config.CONFIDENCE_LEVELS) - 1,
                             format_func=lambda v: f"{v:.0%}",
                             key="hedge_level")
    with top[2]:
        risk_model = st.selectbox("Risk model", (state.GBM, state.BOOTSTRAP),
                                  key="hedge_risk_model",
                                  help="GBM assumes normal daily returns. The "
                                       "bootstrap resamples observed ones.")
    with top[3]:
        objective = st.selectbox("Optimizer objective", OBJECTIVES,
                                 index=3, key="hedge_objective")

    # The sliders are always rendered, and merely disabled when they do not
    # apply. Streamlit clears the session state of a keyed widget that a rerun
    # does not draw, so hiding them would silently reset a viewer's weights to
    # 0.5 every time they looked at another objective and came back.
    balanced = objective == "Balanced"
    left, right, note = st.columns([1, 1, 1.4], gap="small")
    with left:
        st.slider("Protection weight", 0.0, 1.0, step=0.05,
                  key="hedge_protection_weight", on_change=_sync_cost,
                  disabled=not balanced)
    with right:
        st.slider("Cost weight", 0.0, 1.0, step=0.05,
                  key="hedge_cost_weight", on_change=_sync_protection,
                  disabled=not balanced)
    with note:
        st.caption(
            "The two always sum to 1: moving one moves the other. A pair that "
            "did not sum to 1 would rescale the score and make two runs "
            "incomparable."
            if balanced else
            "The weights apply to the Balanced objective. They are kept as you "
            "set them while you look at the others.")

    return (level, risk_model, objective,
            float(st.session_state.hedge_protection_weight),
            float(st.session_state.hedge_cost_weight))


def render():
    snapshot = dl.load_snapshot()
    c.page_header(
        "Which put should I buy?",
        "Compare downside protection against the cost of the insurance.")
    c.timestamp_badge(snapshot, state.get("data_mode"), state.LIVE)

    if snapshot is None or dl.load_table("hedge_candidates") is None:
        c.warning_card(
            "The optimizer has not been run yet",
            "Run <code>python experiments/hedge_optimizer_experiment.py</code> "
            "to price the candidate contracts and build their pricing grids.",
            kind="bad")
        return

    level, risk_model, objective, protection_weight, cost_weight = \
        controls(snapshot)

    if st.button("Analyze hedges", type="primary"):
        with st.spinner(f"Running {config.N_RISK_SCENARIOS:,} scenarios for "
                        f"every candidate under {risk_model}…"):
            st.session_state[TABLE_KEY] = compute.hedge_candidates_table(
                risk_model, config.N_RISK_SCENARIOS,
                config.RISK_HORIZON_DAYS, config.SEED)

    payload = st.session_state.get(TABLE_KEY)
    if payload is None:
        c.callout(
            "Choose an objective and press <b>Analyze hedges</b>. Nothing is "
            f"simulated until you do — each run puts "
            f"{config.N_RISK_SCENARIOS:,} scenarios through every candidate.")
        with st.expander("How to read this page", expanded=True):
            st.markdown(HOW_TO_READ)
        return

    if payload["risk_model"] != risk_model:
        c.warning_card(
            "Showing the previous risk model",
            f"These numbers were computed under <b>{payload['risk_model']}</b>. "
            f"Press <b>Analyze hedges</b> to recompute them under "
            f"<b>{risk_model}</b>.")

    frame, winners = compute.rank_for_objective(
        payload["candidates"], level, protection_weight, cost_weight)
    chosen = compute.objective_winner(frame, winners, objective, level)
    key = f"{level:.0%}".replace("%", "")

    _recommendation(chosen, objective, level, key, payload)
    _frontier(frame, chosen, level, key, payload)
    _supporting_charts(frame, key)
    _table(frame, level)
    _footer(frame, chosen, key, payload, level)


def _recommendation(chosen, objective, level, key, payload):
    st.markdown("#### Best for the selected objective")
    st.caption(f"Objective: **{objective}** · {level:.0%} confidence · "
               f"{payload['risk_model']} · "
               f"{payload['n_scenarios']:,} scenarios")

    c.metric_row([
        ("Recommended strike", strike_label(chosen["strike"]),
         f"{chosen['moneyness']:.1%} of spot, "
         f"{money(payload['spot'])} today"),
        ("Premium cost", money(chosen["premium_cost"]),
         f"ask {money(chosen['ask'])} x "
         f"{config.CONTRACT_MULTIPLIER} shares"),
        ("Hedge cost", percent(chosen["hedge_cost_percent"]),
         "of the share position"),
        (f"{level:.0%} CVaR reduction",
         percent(chosen[f"cvar_{key}_reduction"]),
         f"from {money(payload['baseline'][f'cvar_{key}_dollars'])} "
         f"to {money(chosen[f'cvar_{key}_dollars'])}"),
    ])
    st.write("")
    c.metric_row([
        ("CVaR saved per $1 of premium",
         money(chosen[f"cvar_{key}_saved_per_premium_dollar"]),
         "tail loss avoided for each dollar spent"),
        (f"{level:.0%} VaR reduction",
         percent(chosen[f"var_{key}_reduction"]),
         "where the bad tail begins"),
        ("Behind the unhedged portfolio",
         percent(chosen["worse_than_unhedged_fraction"] * 100),
         "of scenarios — insurance usually costs something"),
        ("Pareto efficient",
         "yes" if bool(chosen["pareto_efficient"]) else "no",
         "nothing available is both cheaper and stronger"),
    ])


def _frontier(frame, chosen, level, key, payload):
    st.markdown("#### The protection-cost frontier")
    st.caption("One point per real listed contract. Hover for its quotes and "
               "every reduction it achieves.")

    hover = frame.assign(
        Strike=frame["strike"].map(lambda v: f"{v:g}"),
        Expiry=frame["expiry"],
        Moneyness=frame["moneyness"].map(lambda v: f"{v:.1%} of spot"),
        Bid=frame["bid"].map(lambda v: f"${v:,.2f}"),
        Ask=frame["ask"].map(lambda v: f"${v:,.2f}"),
        Premium=frame["premium_cost"].map(lambda v: f"${v:,.2f}"),
        Cost=frame["hedge_cost_percent"].map(lambda v: f"{v:.2f}%"),
        VaR_reduction=frame[f"var_{key}_reduction"].map(lambda v: f"{v:.2f}%"),
        CVaR_reduction=frame[f"cvar_{key}_reduction"].map(lambda v: f"{v:.2f}%"),
        Efficiency=frame[f"cvar_{key}_saved_per_premium_dollar"].map(
            lambda v: f"${v:,.2f} per $1"),
    )[["Strike", "Expiry", "Moneyness", "Bid", "Ask", "Premium", "Cost",
       "VaR_reduction", "CVaR_reduction", "Efficiency"]]
    hover.columns = [name.replace("_", " ") for name in hover.columns]

    highlight = int(np.flatnonzero(
        frame["strike"].to_numpy() == chosen["strike"])[0])
    st.plotly_chart(
        ic.frontier(
            frame["hedge_cost_percent"], frame[f"cvar_{key}_reduction"],
            [f"K={v:g}" for v in frame["strike"]],
            pareto=frame["pareto_efficient"].to_numpy(), highlight=highlight,
            xlabel="hedge cost, as a percentage of the share position",
            ylabel=f"{level:.0%} CVaR reduction",
            hover=hover),
        width="stretch", config={"displayModeBar": False})


def _supporting_charts(frame, key):
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("#### Value for money")
        st.plotly_chart(
            ic.efficiency_by_strike(
                frame["strike"],
                {"95% CVaR": frame["cvar_95_saved_per_premium_dollar"],
                 "99% CVaR": frame["cvar_99_saved_per_premium_dollar"]}),
            width="stretch", config={"displayModeBar": False})
        best = frame.loc[frame[f"cvar_{key}_saved_per_premium_dollar"].idxmax()]
        st.caption(
            f"Efficiency peaks at K={best['strike']:g}, not at the cheapest "
            f"strike and not at the strongest. The curve turns over because "
            f"protection and premium do not rise at the same rate.")

    with right:
        st.markdown("#### Protection against strike")
        st.plotly_chart(
            ic.reduction_by_strike(
                frame["moneyness"],
                {"95% VaR": frame["var_95_reduction"],
                 "95% CVaR": frame["cvar_95_reduction"],
                 "99% VaR": frame["var_99_reduction"],
                 "99% CVaR": frame["cvar_99_reduction"]}),
            width="stretch", config={"displayModeBar": False})
        st.caption(
            "CVaR always falls further than VaR: the put works hardest deep in "
            "the tail, which is exactly the region CVaR averages over and VaR "
            "only points at.")


def _table(frame, level):
    st.markdown("#### Every candidate")
    columns = ["strike", "moneyness", "ask", "premium_cost",
               "hedge_cost_percent", "var_95_reduction", "cvar_95_reduction",
               "var_99_reduction", "cvar_99_reduction",
               "cvar_99_saved_per_premium_dollar", "pareto_efficient"]
    st.dataframe(
        frame[columns].sort_values("strike"), hide_index=True, width="stretch",
        column_config={
            "strike": st.column_config.NumberColumn(
                "Strike", format="%.0f", help="The listed strike price."),
            "moneyness": st.column_config.NumberColumn(
                "Moneyness", format="percent",
                help="Strike as a share of the current spot."),
            "ask": st.column_config.NumberColumn(
                "Ask", format="dollar",
                help="What a buyer pays per share. The mid is in the CSV."),
            "premium_cost": st.column_config.NumberColumn(
                "Premium", format="dollar",
                help="Ask times 100 shares -- the whole cost of the hedge."),
            "hedge_cost_percent": st.column_config.NumberColumn(
                "Cost %", format="%.2f%%",
                help="Premium as a percentage of the share position."),
            "var_95_reduction": st.column_config.NumberColumn(
                "95% VaR red.", format="%.2f%%"),
            "cvar_95_reduction": st.column_config.NumberColumn(
                "95% CVaR red.", format="%.2f%%"),
            "var_99_reduction": st.column_config.NumberColumn(
                "99% VaR red.", format="%.2f%%"),
            "cvar_99_reduction": st.column_config.NumberColumn(
                "99% CVaR red.", format="%.2f%%",
                help="Reduction against the same unhedged benchmark, in "
                     "dollars, so the strikes are directly comparable."),
            "cvar_99_saved_per_premium_dollar": st.column_config.NumberColumn(
                "CVaR saved / $", format="dollar",
                help="Dollars of 99% CVaR avoided for each dollar of premium."),
            "pareto_efficient": st.column_config.CheckboxColumn(
                "Pareto efficient",
                help="Nothing else on the board is both cheaper and stronger."),
        })


def _footer(frame, chosen, key, payload, level):
    st.divider()
    if state.explain_simply():
        st.markdown("#### What does this mean?")
        for line in explanations.hedging(
                cvar_reduction=float(chosen[f"cvar_{key}_reduction"]),
                cost_percent=float(chosen["hedge_cost_percent"]),
                saved_per_dollar=float(
                    chosen[f"cvar_{key}_saved_per_premium_dollar"]),
                worse_fraction=float(chosen["worse_than_unhedged_fraction"])):
            st.markdown(f"- {line}")

    with st.expander("How to read this page", expanded=False):
        st.markdown(HOW_TO_READ)

    c.assumption_box([
        ("Cost", "The ask, which is what a buyer pays. Using the mid would "
                 "understate every premium on this page."),
        ("Volatility", "Each contract is calibrated to its own market quote "
                       "against the CRR American tree, so the put skew is "
                       "priced rather than assumed."),
        ("Scenarios", f"{payload['n_scenarios']:,} horizon outcomes over "
                      f"{config.RISK_HORIZON_DAYS} trading days, the same set "
                      f"for every candidate."),
        ("Repricing", "The put is valued at the horizon from a cached pricing "
                      "grid, not by a nested simulation."),
        ("Basis", "Reductions are measured in dollars against one shared "
                  "unhedged benchmark. The percent-of-own-value basis is in "
                  "the saved CSV."),
    ])
