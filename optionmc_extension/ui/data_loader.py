"""Reading the pipeline's saved results, and only then the network.

Presentation mode is the default and this module is why it works: every page
gets its numbers from the CSVs and JSON the batch pipeline already wrote, so
opening the dashboard touches no network at all. Nothing here calls yfinance
unless `refresh_market_data` is invoked, and that only happens behind an
explicit button.

Everything is wrapped in `st.cache_data`, so a file is parsed once per session
rather than once per rerun -- Streamlit re-executes the whole script on every
widget interaction, and re-reading fourteen CSVs each time would make the app
feel broken.

A missing file returns None rather than raising. A viewer who has not run every
phase should still get a working dashboard that tells them which phase to run,
not a traceback on a projector.
"""
import numpy as np
import pandas as pd
import streamlit as st

import config
from src.market_data import MarketSnapshot

SNAPSHOT_JSON = config.DATA_DIR / "market_snapshot.json"
GRID_JSON = config.DATA_DIR / "pricing_grid.json"
PORTFOLIO_LOSSES = config.DATA_DIR / "portfolio_losses.npz"
BOOTSTRAP_LOSSES = config.DATA_DIR / "bootstrap_losses.npz"

# Friendly name -> (file, the experiment that writes it)
TABLES = {
    "baseline_pricing": ("spy_american_put_pricing.csv",
                         "experiments/price_spy_put.py"),
    "convergence": ("experiment1_convergence.csv",
                    "experiments/convergence.py"),
    "discretization": ("experiment2_discretization.csv",
                       "experiments/discretization_test.py"),
    "regression": ("experiment3_regression.csv",
                   "experiments/regression_test.py"),
    "pricing_grid": ("pricing_grid.csv",
                     "experiments/interpolation_test.py"),
    "interpolation_accuracy": ("interpolation_accuracy.csv",
                               "experiments/interpolation_test.py"),
    "exercise_boundary": ("figure08_exercise_boundary.csv",
                          "experiments/make_figures.py"),
    "risk_dollars": ("risk_var_cvar.csv", "experiments/portfolio_risk.py"),
    "risk_percent": ("risk_var_cvar_percent.csv",
                     "experiments/portfolio_risk.py"),
    "portfolio_summary": ("portfolio_loss_summary.csv",
                          "experiments/portfolio_risk.py"),
    "hedge_candidates": ("hedge_optimizer_candidates.csv",
                         "experiments/hedge_optimizer_experiment.py"),
    "hedge_rankings": ("hedge_optimizer_rankings.csv",
                       "experiments/hedge_optimizer_experiment.py"),
    "hedge_frontier": ("protection_cost_frontier.csv",
                       "experiments/hedge_optimizer_experiment.py"),
    "cross_section": ("option_cross_section_snapshot.csv",
                      "experiments/cross_section_validation_experiment.py"),
    "calibration_set": ("calibration_set.csv",
                        "experiments/cross_section_validation_experiment.py"),
    "heldout": ("heldout_predictions.csv",
                "experiments/cross_section_validation_experiment.py"),
    "cross_section_metrics": ("cross_section_metrics.csv",
                              "experiments/cross_section_validation_experiment.py"),
    "spacing_study": ("cross_section_spacing_study.csv",
                      "experiments/cross_section_validation_experiment.py"),
    "risk_models": ("risk_model_comparison.csv",
                    "experiments/bootstrap_risk_experiment.py"),
    "return_diagnostics": ("return_distribution_diagnostics.csv",
                           "experiments/bootstrap_risk_experiment.py"),
    "horizon_quantiles": ("horizon_quantile_comparison.csv",
                          "experiments/bootstrap_risk_experiment.py"),
    "stress": ("stress_test_results.csv",
               "experiments/stress_test_experiment.py"),
    "stress_pricers": ("stress_test_pricer_agreement.csv",
                       "experiments/stress_test_experiment.py"),
}


@st.cache_data(show_spinner=False)
def load_snapshot():
    """The frozen market inputs every page reads from. None if never fetched."""
    if not SNAPSHOT_JSON.exists():
        return None
    try:
        return MarketSnapshot.from_json(SNAPSHOT_JSON)
    except (ValueError, KeyError, TypeError):
        return None


@st.cache_data(show_spinner=False)
def load_table(name):
    """One results table by friendly name, or None if that phase has not run."""
    entry = TABLES.get(name)
    if entry is None:
        return None
    path = config.TABLES_DIR / entry[0]
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, OSError):
        return None


@st.cache_data(show_spinner=False)
def load_losses(which="portfolio"):
    """Loss distributions as a plain dict of arrays.

    Converted out of the lazy NpzFile because `cache_data` pickles what it
    stores, and an open file handle does not survive that.
    """
    path = PORTFOLIO_LOSSES if which == "portfolio" else BOOTSTRAP_LOSSES
    if not path.exists():
        return None
    try:
        with np.load(path) as payload:
            return {key: payload[key] for key in payload.files}
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def load_pricing_grid():
    """The cached American put grid, as spots and prices."""
    if not GRID_JSON.exists():
        return None
    try:
        from src.pricing_grid import load_grid

        grid = load_grid(GRID_JSON)
        return {
            "spots": grid.spots, "prices": grid.prices,
            "strike": grid.strike, "time_to_expiry": grid.time_to_expiry,
            "volatility": grid.volatility, "n_paths": grid.n_paths,
            "degree": grid.degree,
        }
    except (OSError, ValueError, KeyError):
        return None


def availability():
    """What is on disk and what is missing, with the command that fixes it.

    Drives the data-source panel, and lets a page say "run this" instead of
    rendering an empty chart.
    """
    rows = [{
        "item": "market snapshot",
        "present": SNAPSHOT_JSON.exists(),
        "produced_by": "experiments/fetch_market_data.py",
    }, {
        "item": "pricing grid",
        "present": GRID_JSON.exists(),
        "produced_by": "experiments/portfolio_risk.py",
    }, {
        "item": "portfolio losses",
        "present": PORTFOLIO_LOSSES.exists(),
        "produced_by": "experiments/portfolio_risk.py",
    }, {
        "item": "bootstrap losses",
        "present": BOOTSTRAP_LOSSES.exists(),
        "produced_by": "experiments/bootstrap_risk_experiment.py",
    }]
    for name, (filename, producer) in TABLES.items():
        rows.append({
            "item": name.replace("_", " "),
            "present": (config.TABLES_DIR / filename).exists(),
            "produced_by": producer,
        })
    return pd.DataFrame(rows)


def missing_items():
    """Just the gaps, for a warning banner."""
    frame = availability()
    return frame[~frame["present"]]


def figure_path(name):
    """A publication figure by stem, if it was generated."""
    path = config.FIGURES_DIR / f"{name}.png"
    return path if path.exists() else None


def clear_caches():
    """Drop every cached read. Called after a live refresh rewrites the files."""
    load_snapshot.clear()
    load_table.clear()
    load_losses.clear()
    load_pricing_grid.clear()


def refresh_market_data():
    """Fetch fresh market data. The only function here that touches a network.

    Never raises: a failed download in front of a classroom must leave the
    cached snapshot in place and produce a sentence, not a traceback.

    Returns
    -------
    (snapshot, error)
        `snapshot` is the new MarketSnapshot on success, otherwise None, and
        `error` carries a message the UI can show.
    """
    try:
        from src.market_data import build_market_snapshot

        snapshot = build_market_snapshot(config, force_refresh=True)
        snapshot.to_json(SNAPSHOT_JSON)
        clear_caches()
        return snapshot, None
    except Exception as exc:                       # noqa: BLE001 - see docstring
        return None, f"{type(exc).__name__}: {exc}"
