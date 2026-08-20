"""Methodology, sources and how to run all of it again.

Kept short on purpose. This page says what was assumed, where the data came
from, what the parameters were, and the exact commands to reproduce every
number -- not a second copy of the report.
"""
import streamlit as st

import config
from ui import components as c
from ui import compute
from ui import data_loader as dl
from ui import state
from ui.formatters import count, money, percent, volatility

TITLE = ("Extending OptionMC: American option pricing with Least-Squares "
         "Monte Carlo and portfolio risk analysis using VaR and CVaR")

REFERENCES = [
    ("Base paper",
     "OptionMC: A Python Package for Monte Carlo Pricing of European Options",
     "Herho and others, 2025",
     "Reproduced first and left untouched. Its European Monte Carlo is the "
     "starting point everything here extends."),
    ("Main extension",
     "Valuing American Options by Simulation: A Simple Least-Squares Approach",
     "Longstaff and Schwartz, 2001",
     "The method that makes early exercise tractable in a simulation. This "
     "project implements it rather than calling a library, and checks it "
     "against the paper's own worked example and Table 1."),
    ("Risk reference",
     "Optimization of Conditional Value-at-Risk",
     "Rockafellar and Uryasev, 2000",
     "CVaR and the minimisation form used to cross-check it. The dashboard "
     "reports both VaR and CVaR because the second sees what the first "
     "cannot."),
]

ASSUMPTIONS = [
    ("Underlying dynamics",
     "Geometric Brownian motion with constant volatility. No jumps, no "
     "stochastic volatility, and no volatility term structure."),
    ("Measure separation",
     "Option pricing uses the risk-neutral drift r minus q. Portfolio risk "
     "uses the historical drift mu. These never mix, and an invariant on the "
     "numerical methods page checks it."),
    ("Exercise",
     f"American exercise is approximated by {config.LSMC_N_STEPS} equally "
     f"spaced opportunities, which makes the contract formally Bermudan and "
     f"the price a slight under-estimate. The size of that gap is measured "
     f"rather than assumed."),
    ("Costs",
     "The ask is treated as the full cost of buying protection. Commissions, "
     "financing and slippage beyond the quoted spread are not modelled."),
    ("Independence",
     "The historical bootstrap draws days independently, which reproduces fat "
     "tails but discards volatility clustering -- so a multi-day crash is "
     "less likely under it than in the market."),
    ("Static hedge",
     "The put is bought once and held to the horizon. No rebalancing, and no "
     "early exercise of the hedge itself."),
]

LIMITATIONS = [
    "GBM is a simplified market model, and the bootstrap can only resample "
    "the history it was given.",
    "Market quotes can be stale or illiquid; the snapshot is a single moment "
    "and options are not equally traded.",
    "One expiry has live quotes in the cached chain, so the cross-section "
    "validates across strikes rather than across maturities.",
    "Model parameters drift through time; a volatility estimated over five "
    "years is not a forecast of the next ten days.",
    "Every result is conditional on the frozen snapshot below. Refreshing the "
    "market data will move all of them.",
]


def render():
    snapshot = dl.load_snapshot()
    c.page_header("Methodology and about",
                  "What was assumed, where the data came from, and how to run "
                  "it again.")
    c.timestamp_badge(snapshot, state.get("data_mode"), state.LIVE)

    st.markdown(f"##### {TITLE}")
    st.caption("CSE402 Numerical Analysis, Simulation and Modeling")

    _references()
    _data(snapshot)
    _parameters(snapshot)
    _assumptions()
    _reproducibility()
    _limitations()


def _references():
    st.markdown("#### What this is built on")
    for column, entry in zip(st.columns(3, gap="small"), REFERENCES):
        role, title, authors, note = entry
        with column:
            c.metric_card(role, title, f"{authors}<br><br>{note}", accent=True)
    st.write("")


def _data(snapshot):
    st.markdown("#### Where the data comes from")
    rows = [
        ("Underlying", f"{config.TICKER} ETF. Chosen because its listed "
                       f"options are American-style; SPX and XSP are European "
                       f"and would defeat the purpose."),
        ("Option chain", "Yahoo Finance via yfinance, cached to "
                         "data/spy_option_snapshot.csv."),
        ("Risk-free rate", f"FRED {config.FRED_SERIES}, the 3-month Treasury "
                           f"constant maturity rate, converted to a "
                           f"continuously compounded rate."),
        ("Dividend yield", "Estimated from the trailing twelve months of "
                           "distributions in the cached price history."),
        ("Price history", f"{config.HISTORY_PERIOD} of daily closes, used for "
                          f"the real-world drift and volatility and resampled "
                          f"by the bootstrap."),
    ]
    if snapshot is not None:
        rows.insert(0, (
            "Snapshot", f"{snapshot.as_of} · spot {money(snapshot.spot)} · "
                        f"{snapshot.n_history_days} trading days of history "
                        f"({snapshot.history_start} to "
                        f"{snapshot.history_end}) · risk-free rate dated "
                        f"{snapshot.risk_free_date}"))
    for label, detail in rows:
        st.markdown(f"**{label}** — {detail}")
    st.write("")


def _parameters(snapshot):
    st.markdown("#### The parameters everything was run with")
    items = [
        ("Ticker", config.TICKER, "American-style listed options"),
        ("Option style", "American put", "early exercise permitted"),
        ("Contract multiplier", count(config.CONTRACT_MULTIPLIER),
         f"one contract covers {config.SHARES} shares"),
        ("Risk horizon", f"{config.RISK_HORIZON_DAYS} trading days",
         f"{config.RISK_HORIZON_DAYS / config.TRADING_DAYS_PER_YEAR:.4f} years"),
    ]
    c.metric_row(items)
    st.write("")
    c.metric_row([
        ("Scenarios", count(config.N_RISK_SCENARIOS), "per risk model"),
        ("Random seed", count(config.SEED),
         "every experiment seeds from this"),
        ("LSMC paths / steps",
         f"{config.LSMC_N_PATHS:,} / {config.LSMC_N_STEPS}",
         f"degree {config.LSMC_DEGREE} basis"),
        ("Pricing grid",
         f"{config.GRID_N_POINTS} nodes",
         f"{config.GRID_N_PATHS:,} paths, degree {config.GRID_DEGREE}, "
         f"{config.INTERPOLATION_METHOD}"),
    ])

    if snapshot is not None:
        st.write("")
        c.metric_row([
            ("Historical volatility",
             volatility(snapshot.historical_volatility), "annualised sigma"),
            ("Historical drift", percent(snapshot.historical_drift * 100),
             "annualised mu, real world only"),
            ("Risk-free rate", percent(snapshot.risk_free_rate * 100),
             "continuously compounded"),
            ("Dividend yield", percent(snapshot.dividend_yield * 100),
             "continuous q"),
        ])
    st.write("")


def _assumptions():
    st.markdown("#### What had to be true")
    for label, detail in ASSUMPTIONS:
        st.markdown(f"**{label}** — {detail}")
    st.write("")


def _reproducibility():
    facts = compute.project_facts()

    st.markdown("#### Reproducing all of it")
    c.metric_row([
        ("Test functions", count(facts["test_functions"]),
         f"across {facts['test_files']} files"),
        ("Numerical modules", count(facts["source_modules"]),
         f"plus {facts['experiments']} experiment scripts"),
        ("Saved results", f"{facts['tables']} tables, {facts['figures']} "
                          f"figures",
         f"last written {facts['results_written'] or 'never'}"),
        ("Dashboard modules", count(facts["ui_modules"]),
         "none of which implement mathematics"),
    ])
    st.caption(
        "Counted by parsing the repository, not written down here. The figure "
        "above counts test *functions*; parametrised ones expand into several "
        "cases each, so the suite reports a larger number when it runs.")

    st.markdown("**From the `optionmc_extension` directory:**")
    st.code(
        "..\\.venv\\Scripts\\python -m pytest            # the whole test suite\n"
        "..\\.venv\\Scripts\\python main.py --skip-fetch  # every phase, cached data\n"
        "..\\.venv\\Scripts\\python main.py               # the same, refreshing market data\n"
        "..\\.venv\\Scripts\\streamlit run app.py         # this dashboard",
        language="text")
    st.caption("Windows paths. On Linux or macOS replace the backslashes with "
               "forward slashes and `Scripts` with `bin`.")

    c.callout(
        "<b>Presentation mode works without live internet.</b> The dashboard "
        "opens on the cached snapshot and reads only saved results, so nothing "
        "on it depends on a network. Live data is fetched only when the mode "
        "is switched and the refresh button is pressed, and a failed fetch "
        "leaves the cached numbers in place.", kind="good")
    st.write("")


def _limitations():
    st.markdown("#### What this does not claim")
    for line in LIMITATIONS:
        st.markdown(f"- {line}")

    c.callout(
        "<b>This is an educational numerical-analysis project.</b> It is not "
        "financial advice, not a recommendation to trade, and not a "
        "production risk system. Every number on it is conditional on a single "
        "frozen market snapshot and on the model assumptions listed above.",
        kind="warn")
