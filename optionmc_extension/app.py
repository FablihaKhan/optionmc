#!/usr/bin/env python
"""OptionMC Advanced Risk Lab -- the dashboard entry point.

    streamlit run app.py

Opens in presentation mode: every number comes from the results the pipeline
already wrote to disk, so the app needs no internet and loads immediately. Live
market data is fetched only when someone switches mode and presses the refresh
button. Nothing on this page calls yfinance on load, which is the difference
between a demonstration that works in a classroom and one that depends on the
room's wifi.

This file wires the app together and owns no mathematics. Pages live in
`ui_pages/`, shared machinery in `ui/`, and every number ultimately comes from
`src/`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

import config
from ui import components as c
from ui import data_loader as dl
from ui import state
from ui_pages import (about, hedge_optimizer, home, market_validation,
                      numerical_methods, pricing_lab, risk_lab)

TITLE = "OptionMC Advanced Risk Lab"
SUBTITLE = "American Option Pricing, Hedging and Tail-Risk Decision Support"
CAPTION = "CSE402 Numerical Analysis, Simulation and Modeling"


def sidebar(snapshot):
    """Data mode, the shared parameters, and the explain toggle.

    Every control here is bound to a session-state key, so a viewer's choice
    survives moving between pages. None of them triggers a calculation: the
    expensive work sits behind each page's own run button, so dragging a
    slider can never start fifty thousand simulations.
    """
    with st.sidebar:
        st.markdown(f"### {TITLE}")
        st.caption(CAPTION)
        st.divider()

        st.markdown("**Data mode**")
        st.radio(
            "Data mode", (state.CACHED, state.LIVE), key="data_mode",
            label_visibility="collapsed",
            help="Cached is the default and needs no internet. Live fetches "
                 "fresh market data only when you press refresh.")

        if snapshot is not None:
            from ui.formatters import snapshot_caption
            st.caption(snapshot_caption(snapshot.as_of, snapshot.ticker,
                                        snapshot.days_to_expiry))

        if state.get("data_mode") == state.LIVE:
            if st.button("Refresh live market data", width="stretch"):
                with st.spinner("Fetching from Yahoo Finance and FRED…"):
                    fresh, error = dl.refresh_market_data()
                if error:
                    # The previous snapshot is still on disk and still on
                    # screen; a failed fetch must not empty the dashboard.
                    state.set_value("refresh_error", error)
                    state.set_value("last_refresh", None)
                else:
                    state.set_value("refresh_error", None)
                    state.set_value("last_refresh", fresh.as_of)
                    state.set_value("expiry", fresh.expiry)
                    state.set_value("strike", fresh.strike)
                st.rerun()

            if state.get("refresh_error"):
                st.warning("Live refresh failed. The cached snapshot is still "
                           "in use, so everything on screen remains valid.")
                with st.expander("Technical detail"):
                    st.code(state.get("refresh_error"))
            elif state.get("last_refresh"):
                st.success(f"Refreshed to {state.get('last_refresh')}")
        else:
            st.caption("Presentation mode: reading saved results only. "
                       "No network access.")

        st.divider()
        st.toggle(
            "Explain simply", key="explain_simply",
            help="Adds a short plain-English reading under each set of "
                 "numbers, generated from the calculated values.")

        st.divider()
        with st.expander("Simulation parameters", expanded=False):
            st.number_input("Monte Carlo paths", min_value=1_000,
                            max_value=200_000, step=1_000, key="n_paths")
            st.number_input("Exercise steps", min_value=5, max_value=250,
                            step=5, key="n_steps")
            st.number_input("Regression degree", min_value=1, max_value=4,
                            step=1, key="degree")
            st.number_input("Random seed", min_value=0, max_value=10_000,
                            step=1, key="seed")
            st.caption("These are the pipeline's defaults. Changing them here "
                       "affects the pricing lab; the saved results keep the "
                       "values they were computed with.")

        with st.expander("Risk settings", expanded=False):
            st.selectbox("Confidence level", (0.95, 0.99),
                         key="confidence_level",
                         format_func=lambda v: f"{v:.0%}")
            st.selectbox("Risk model", (state.GBM, state.BOOTSTRAP),
                         key="risk_model")
            st.caption(f"{config.N_RISK_SCENARIOS:,} scenarios over "
                       f"{config.RISK_HORIZON_DAYS} trading days.")

        st.divider()
        if st.button("Reset to defaults", width="stretch"):
            state.reset()
            st.rerun()

        missing = dl.missing_items()
        if len(missing):
            st.caption(f"{len(missing)} result files are not on disk. "
                       "Run `python main.py --skip-fetch` to build them.")


def main():
    st.set_page_config(page_title=TITLE, page_icon="📉", layout="wide",
                       initial_sidebar_state="expanded")
    c.inject_css()

    snapshot = dl.load_snapshot()
    state.initialise(snapshot)
    sidebar(snapshot)

    c.hero(TITLE, SUBTITLE, CAPTION)

    # Every page module exposes `render`, so Streamlit would infer the same URL
    # for all seven and refuse to build the navigation. The paths are therefore
    # given explicitly, which also keeps them stable and linkable.
    pages = [
        st.Page(home.render, title="Overview", icon="🏠",
                url_path="overview", default=True),
        st.Page(pricing_lab.render, title="Pricing Lab", icon="📈",
                url_path="pricing-lab"),
        st.Page(hedge_optimizer.render, title="Hedge Optimizer", icon="🛡️",
                url_path="hedge-optimizer"),
        st.Page(market_validation.render, title="Market Validation", icon="🎯",
                url_path="market-validation"),
        st.Page(risk_lab.render, title="Risk & Stress Lab", icon="⚠️",
                url_path="risk-lab"),
        st.Page(numerical_methods.render, title="Numerical Methods", icon="🧮",
                url_path="numerical-methods"),
        st.Page(about.render, title="Methodology & About", icon="📚",
                url_path="about"),
    ]
    st.navigation(pages, position="sidebar").run()

    c.footer()


main()
