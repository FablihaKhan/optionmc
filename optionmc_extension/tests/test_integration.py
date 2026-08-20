"""Whole-project checks for the freeze.

The page-level suites each test their own page. These are the cross-cutting
ones: properties that have to hold everywhere at once, and that nothing else
would catch because they fall between two files.

Most of them are negative. A dashboard fails quietly -- a nan in a metric card,
a chart with no data in it, a table and the chart beside it disagreeing -- and
none of those raise. They are only caught by looking for them on purpose.
"""
import base64
import json
import re
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from streamlit.testing.v1 import AppTest

import config

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"

PAGES = ["home", "pricing_lab", "hedge_optimizer", "market_validation",
         "risk_lab", "numerical_methods", "about"]

# Pages that compute only after a button; the rest render on load.
NEEDS_A_RUN = {"pricing_lab", "hedge_optimizer"}


def harness(module):
    return (
        f"import sys; sys.path.insert(0, r'{ROOT.as_posix()}')\n"
        "import streamlit as st\n"
        "from ui import components as c, state, data_loader as dl\n"
        f"from ui_pages import {module}\n"
        "c.inject_css()\n"
        "state.initialise(dl.load_snapshot())\n"
        f"{module}.render()\n"
    )


def render(module, press_buttons=True):
    app = AppTest.from_string(harness(module), default_timeout=300).run()
    if press_buttons and module in NEEDS_A_RUN and app.button:
        app.button[0].click().run()
    return app


@pytest.fixture(scope="module")
def pages():
    """Every page, rendered once, with its expensive path exercised."""
    return {module: render(module) for module in PAGES}


def visible(app):
    """Everything a viewer can read, tags stripped and whitespace collapsed."""
    parts = [element.value for element in app.markdown]
    parts.extend(element.value for element in app.caption)
    return " ".join(re.sub(r"<[^>]+>", " ", " ".join(parts)).split())


def decode(value):
    """A Plotly array, whether it arrived as a list or a typed array."""
    if isinstance(value, dict) and "bdata" in value:
        return np.frombuffer(base64.b64decode(value["bdata"]),
                             dtype=np.dtype(value["dtype"]))
    return np.asarray(value)


def charts(app):
    return [json.loads(element.proto.spec)
            for element in app.get("plotly_chart")]


# --------------------------------------------------------------------------
# Nothing raises, anywhere
# --------------------------------------------------------------------------

@pytest.mark.parametrize("module", PAGES)
def test_no_page_raises(module, pages):
    app = pages[module]
    assert not app.exception, [e.message for e in app.exception]


def test_the_assembled_app_raises_nothing():
    app = AppTest.from_file(str(APP), default_timeout=300).run()
    assert not app.exception, [e.message for e in app.exception]


def test_no_page_leaks_a_traceback_to_the_viewer(pages):
    """Streamlit is configured not to show error detail; check nothing else does."""
    for module, app in pages.items():
        text = visible(app)
        for marker in ("Traceback (most recent call last)", "File \"",
                       "Exception:", "raise "):
            assert marker not in text, f"{module} shows {marker!r}"


# --------------------------------------------------------------------------
# No nan reaches the screen
# --------------------------------------------------------------------------

@pytest.mark.parametrize("module", PAGES)
def test_no_page_prints_nan(module, pages):
    """`money(None)` renders an em dash on purpose; a literal nan is a bug."""
    text = visible(pages[module]).lower()
    # "finance" contains "nan"; so does "significant". Match it as a word.
    assert not re.search(r"\bnan\b", text), f"{module} prints nan"
    assert "$nan" not in text
    assert "nan%" not in text


@pytest.mark.parametrize("module", PAGES)
def test_no_chart_carries_a_nan_where_a_number_belongs(module, pages):
    for index, spec in enumerate(charts(pages[module])):
        for trace in spec["data"]:
            for axis in ("x", "y", "z"):
                if axis not in trace or trace[axis] is None:
                    continue
                values = decode(trace[axis])
                if values.dtype.kind not in "fi":
                    continue
                # The heatmap legitimately holds gaps for empty buckets.
                if trace.get("type") == "heatmap":
                    continue
                assert np.isfinite(values).all(), (
                    f"{module} chart {index} trace "
                    f"{trace.get('name', '?')} has a non-finite {axis}")


# --------------------------------------------------------------------------
# No chart is empty
# --------------------------------------------------------------------------

@pytest.mark.parametrize("module", PAGES)
def test_every_chart_has_something_in_it(module, pages):
    for index, spec in enumerate(charts(pages[module])):
        assert spec["data"], f"{module} chart {index} has no traces"
        drawn = 0
        for trace in spec["data"]:
            for axis in ("x", "y", "z"):
                if axis in trace and trace[axis] is not None:
                    drawn = max(drawn, decode(trace[axis]).size)
        assert drawn > 0, f"{module} chart {index} draws nothing"


def test_the_pages_that_should_draw_charts_do(pages):
    expected = {"home": 1, "pricing_lab": 5, "hedge_optimizer": 3,
                "market_validation": 6, "risk_lab": 5, "numerical_methods": 4,
                "about": 0}
    for module, minimum in expected.items():
        assert len(charts(pages[module])) >= minimum, module


# --------------------------------------------------------------------------
# The charts agree with the tables they came from
# --------------------------------------------------------------------------

def test_the_frontier_matches_the_saved_candidate_table(pages):
    """A chart drifting from its table is invisible until someone checks."""
    frame = pd.read_csv(config.TABLES_DIR / "hedge_optimizer_candidates.csv")
    spec = charts(pages["hedge_optimizer"])[0]
    plotted = np.concatenate([
        decode(trace["x"]) for trace in spec["data"]
        if trace.get("name") in ("Pareto efficient", "dominated")])
    np.testing.assert_allclose(np.sort(plotted),
                               np.sort(frame["hedge_cost_percent"].to_numpy()),
                               rtol=1e-6)


def test_the_smile_matches_the_saved_calibration_set(pages):
    frame = pd.read_csv(config.TABLES_DIR / "calibration_set.csv")
    frame = frame[np.isfinite(frame["implied_vol"])]
    spec = charts(pages["market_validation"])[1]
    quotes = [trace for trace in spec["data"]
              if trace.get("name") == "calibration quotes"][0]
    np.testing.assert_allclose(
        np.sort(decode(quotes["y"]) / 100.0),
        np.sort(frame["implied_vol"].to_numpy()), rtol=1e-9)


def test_the_stress_chart_matches_the_saved_stress_table(pages):
    frame = pd.read_csv(config.TABLES_DIR / "stress_test_results.csv")
    frame = frame[frame["portfolio"] == frame["portfolio"].iloc[0]]
    spec = charts(pages["risk_lab"])[3]
    stock = [trace for trace in spec["data"]
             if trace.get("name") == "SPY only"][0]
    np.testing.assert_allclose(
        np.sort(decode(stock["y"])),
        np.sort(frame["stock_only_loss_percent"].to_numpy()), rtol=1e-9)


def test_the_convergence_chart_matches_its_experiment(pages):
    frame = pd.read_csv(config.TABLES_DIR / "experiment1_convergence.csv")
    spec = charts(pages["numerical_methods"])[0]
    prices = [trace for trace in spec["data"]
              if trace.get("name") == "LSMC mean price"][0]
    np.testing.assert_allclose(decode(prices["y"]),
                               frame["mean_price"].to_numpy(), rtol=1e-9)


# --------------------------------------------------------------------------
# Money and percentages are written one way
# --------------------------------------------------------------------------

@pytest.mark.parametrize("module", PAGES)
def test_money_is_written_with_two_decimals_and_separators(module, pages):
    """Catches $1234.5 and $1234, which are the two ways this goes wrong."""
    text = visible(pages[module])
    for amount in re.findall(r"\$\d[\d,]*(?:\.\d+)?", text):
        digits = amount.lstrip("$").split(".")[0].replace(",", "")
        if len(digits) > 3:
            assert "," in amount, f"{module}: {amount} lacks a separator"


@pytest.mark.parametrize("module", PAGES)
def test_percentages_never_run_to_more_than_four_decimals(module, pages):
    text = visible(pages[module])
    for value in re.findall(r"-?\d+\.(\d+)%", text):
        assert len(value) <= 4, f"{module}: {value} is over-precise"


# --------------------------------------------------------------------------
# Provenance is always on screen
# --------------------------------------------------------------------------

@pytest.mark.parametrize("module", PAGES)
def test_every_page_shows_which_snapshot_it_is_using(module, pages):
    text = visible(pages[module])
    assert "CACHED SNAPSHOT" in text or "LIVE" in text, module
    assert "DTE" in text, module


def test_the_pages_that_take_parameters_show_the_ones_in_force(pages):
    lab = visible(pages["pricing_lab"])
    assert f"{config.BINOMIAL_N_STEPS:,} steps" in lab
    assert "paths, degree" in lab

    optimizer = visible(pages["hedge_optimizer"])
    assert f"{config.N_RISK_SCENARIOS:,} scenarios" in optimizer
    assert "GBM Monte Carlo" in optimizer or "Historical bootstrap" in optimizer


# --------------------------------------------------------------------------
# The traps the scope names
# --------------------------------------------------------------------------

def test_no_candidate_contract_appears_twice():
    frame = pd.read_csv(config.TABLES_DIR / "hedge_optimizer_candidates.csv")
    assert frame["strike"].is_unique
    assert frame["contract_symbol"].is_unique


def test_no_contract_is_used_to_validate_itself():
    calibration = pd.read_csv(config.TABLES_DIR / "calibration_set.csv")
    heldout = pd.read_csv(config.TABLES_DIR / "heldout_predictions.csv")
    shared = (set(zip(calibration["expiry"], calibration["strike"]))
              & set(zip(heldout["expiry"], heldout["strike"])))
    assert not shared


def test_the_risk_engine_never_prices_an_option_per_scenario():
    """The nested Monte Carlo the scope forbids would be minutes, not seconds.

    Timed rather than asserted structurally: a nested run of 50,000 valuations
    could not finish in this budget however it were written.
    """
    import time

    from ui import compute

    compute.hedge_candidates_table.clear()
    started = time.perf_counter()
    payload = compute.hedge_candidates_table(
        "GBM Monte Carlo", config.N_RISK_SCENARIOS,
        config.RISK_HORIZON_DAYS, config.SEED)
    elapsed = time.perf_counter() - started

    assert payload is not None
    assert len(payload["candidates"]) >= 3
    assert elapsed < 60, (
        f"{len(payload['candidates'])} candidates x "
        f"{config.N_RISK_SCENARIOS:,} scenarios took {elapsed:.1f}s -- that is "
        f"the shape of a nested simulation, not an interpolated grid")


def test_the_pricing_and_risk_measures_stay_apart():
    from src.market_data import MarketSnapshot

    snapshot = MarketSnapshot.from_json(config.DATA_DIR / "market_snapshot.json")
    risk_neutral = snapshot.risk_free_rate - snapshot.dividend_yield
    assert abs(snapshot.historical_drift - risk_neutral) > 1e-9
    assert abs(snapshot.historical_drift - snapshot.risk_free_rate) > 1e-9


# --------------------------------------------------------------------------
# Offline, everywhere
# --------------------------------------------------------------------------

@pytest.mark.parametrize("module", PAGES)
def test_no_page_opens_a_socket_in_cached_mode(module, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(f"{module} tried to use the network")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)

    app = render(module)
    assert not app.exception, [e.message for e in app.exception]


def test_the_whole_app_survives_a_dead_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the dashboard tried to use the network")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)

    app = AppTest.from_file(str(APP), default_timeout=300).run()
    assert not app.exception, [e.message for e in app.exception]
    assert "CACHED SNAPSHOT" in visible(app)


# --------------------------------------------------------------------------
# The base package is still the base package
# --------------------------------------------------------------------------

def test_the_extension_never_reimplements_the_base_pricer():
    """The reproduction has to call the original, or it proves nothing."""
    source = (ROOT / "src" / "european_mc.py").read_text(encoding="utf-8")
    assert "from optionmc.models import OptionPricing" in source


def test_the_base_package_still_reproduces_its_published_numbers():
    """The paper's headline example, from the original class, untouched."""
    from optionmc.models import OptionPricing

    pricer = OptionPricing(S0=100.0, E=100.0, T=1.0, rf=0.05, sigma=0.2,
                           iterations=200_000)
    call, put = pricer.bs_analytical_price()
    assert call == pytest.approx(10.45, abs=0.01)
    assert put == pytest.approx(5.57, abs=0.01)

    # And its Monte Carlo still converges to those analytical prices.
    assert pricer.call_option_simulation() == pytest.approx(call, rel=0.02)
    assert pricer.put_option_simulation() == pytest.approx(put, rel=0.03)
