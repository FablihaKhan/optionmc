"""Smoke tests for the Streamlit dashboard.

The failure these guard against is specific and expensive: the app raising on a
projector, in a room whose wifi does not work, ten seconds into a viva. So the
tests run the real app headlessly through Streamlit's own AppTest harness, load
every page, and assert that a cached-mode run opens no network connection at
all.

They are not a substitute for looking at the thing. They catch the errors that
are invisible until the moment they matter.
"""
import socket
import sys
from pathlib import Path

import pandas as pd
import pytest

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"

PAGES = ["home", "pricing_lab", "hedge_optimizer", "market_validation",
         "risk_lab", "numerical_methods", "about"]

# The heading a page prints, which is its question rather than its navigation
# label. The sidebar still says "Hedge Optimizer"; the page itself asks the
# question it answers.
PAGE_TITLES = {
    "home": "Overview",
    "pricing_lab": "Pricing Lab",
    "hedge_optimizer": "Which put should I buy?",
    "market_validation": "Does the model generalise?",
    "risk_lab": "How bad can the loss get?",
    "numerical_methods": "Inside the numerical engine",
    "about": "Methodology and about",
}


def page_harness(module, extra=""):
    """A minimal script that renders one page the way `app.py` would.

    `st.navigation` builds its pages from callables, which AppTest's
    `switch_page` cannot address -- it resolves file paths under the main
    script. Rendering the module directly exercises the same code with the same
    shared state.
    """
    return (
        f"import sys; sys.path.insert(0, r'{ROOT.as_posix()}')\n"
        "import streamlit as st\n"
        "from ui import components as c, state, data_loader as dl\n"
        f"from ui_pages import {module}\n"
        "c.inject_css()\n"
        "state.initialise(dl.load_snapshot())\n"
        f"{extra}"
        f"{module}.render()\n"
    )


def messages(app):
    return [element.value for element in app.markdown]


@pytest.fixture(scope="module")
def app():
    result = AppTest.from_file(str(APP), default_timeout=120).run()
    return result


# --------------------------------------------------------------------------
# The app itself
# --------------------------------------------------------------------------

def test_the_app_launches_without_raising(app):
    assert not app.exception, [e.message for e in app.exception]


def test_the_hero_carries_the_project_title(app):
    text = " ".join(messages(app))
    assert "OptionMC Advanced Risk Lab" in text
    assert "American Option Pricing, Hedging and Tail-Risk Decision Support" in text


def test_it_opens_in_presentation_mode(app):
    """Cached must be the default, or a dead network breaks the first load."""
    from ui import state

    assert app.radio[0].value == state.CACHED
    assert app.session_state["data_mode"] == state.CACHED


def test_the_data_mode_control_offers_both_modes(app):
    from ui import state

    assert list(app.radio[0].options) == [state.CACHED, state.LIVE]


def test_explain_simply_exists_and_starts_off(app):
    assert len(app.toggle) == 1
    assert app.toggle[0].value is False


def test_the_shared_parameters_are_seeded_from_config(app):
    import config

    assert app.session_state["n_paths"] == config.LSMC_N_PATHS
    assert app.session_state["n_steps"] == config.LSMC_N_STEPS
    assert app.session_state["seed"] == config.SEED
    assert app.session_state["ticker"] == config.TICKER


def test_the_snapshot_badge_is_on_screen(app):
    text = " ".join(messages(app))
    assert "CACHED SNAPSHOT" in text
    assert "DTE" in text


def test_the_footer_states_it_is_not_advice(app):
    assert any("Not financial advice" in value for value in messages(app))


# --------------------------------------------------------------------------
# Every page renders
# --------------------------------------------------------------------------

@pytest.mark.parametrize("module", PAGES)
def test_each_page_renders_without_raising(module):
    result = AppTest.from_string(page_harness(module), default_timeout=120).run()
    assert not result.exception, [e.message for e in result.exception]


@pytest.mark.parametrize("module", PAGES)
def test_each_page_shows_its_own_heading(module):
    result = AppTest.from_string(page_harness(module), default_timeout=120).run()
    assert any(PAGE_TITLES[module] in value for value in messages(result))


def test_the_overview_renders_numbers_from_the_cached_pipeline():
    result = AppTest.from_string(page_harness("home"), default_timeout=120).run()
    text = " ".join(messages(result))
    for section in ("Market snapshot", "Numerical engine",
                    "Portfolio protection", "Tail-risk reduction"):
        assert section in text
    # A real dollar figure, not a placeholder or a nan.
    assert "$" in text
    assert "nan" not in text.lower()


def test_explain_simply_adds_reading_to_the_overview():
    plain = AppTest.from_string(page_harness("home"), default_timeout=120).run()
    explained = AppTest.from_string(
        page_harness("home", extra="state.set_value('explain_simply', True)\n"),
        default_timeout=120).run()
    assert not explained.exception
    assert len(explained.expander) > len(plain.expander)


# --------------------------------------------------------------------------
# Presentation mode is genuinely offline
# --------------------------------------------------------------------------

def test_a_cached_run_never_imports_yfinance():
    """Market data is imported lazily inside the download functions.

    If that ever changes, opening the dashboard would pull in a network client
    at import time, and the first failure would be in front of an audience.
    """
    sys.modules.pop("yfinance", None)
    result = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not result.exception
    assert "yfinance" not in sys.modules


def test_a_cached_run_opens_no_socket(monkeypatch):
    """The decisive version: make any outbound connection fail loudly."""
    def refuse(*args, **kwargs):
        raise AssertionError("presentation mode tried to use the network")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)

    result = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not result.exception, [e.message for e in result.exception]


# --------------------------------------------------------------------------
# Degrading gracefully
# --------------------------------------------------------------------------

def test_the_overview_says_what_to_run_when_the_snapshot_is_missing(monkeypatch):
    """A missing file must produce a sentence, never a traceback.

    Patched through `monkeypatch` rather than inside the harness script.
    AppTest runs in this process and imports the very same `ui.data_loader`
    module, so assigning to it from the script would leave every later test
    looking at an app with no snapshot -- which is exactly what happened.
    """
    from ui import data_loader

    monkeypatch.setattr(data_loader, "load_snapshot", lambda: None)
    result = AppTest.from_string(page_harness("home"), default_timeout=120).run()
    assert not result.exception
    assert any("No market snapshot" in value for value in messages(result))
    assert any("fetch_market_data.py" in value for value in messages(result))


def test_a_failed_live_refresh_returns_a_message_rather_than_raising(monkeypatch):
    from ui import data_loader

    def explode(*args, **kwargs):
        raise ConnectionError("no route to host")

    monkeypatch.setattr("src.market_data.build_market_snapshot", explode)
    snapshot, error = data_loader.refresh_market_data()
    assert snapshot is None
    assert "ConnectionError" in error and "no route to host" in error


# --------------------------------------------------------------------------
# The executive overview (phase 5A)
# --------------------------------------------------------------------------

OVERVIEW_SECTIONS = [
    "The problem, in three steps",
    "Market snapshot",
    "Numerical engine",
    "Portfolio protection",
    "Tail-risk reduction",
    "Which put would you buy?",
    "The argument in one row",
    "Does the answer depend on the risk model?",
    "The project in one sentence",
]


@pytest.fixture(scope="module")
def overview():
    return AppTest.from_string(page_harness("home"), default_timeout=180).run()


@pytest.mark.parametrize("section", OVERVIEW_SECTIONS)
def test_the_overview_carries_every_section(overview, section):
    assert any(section in value for value in messages(overview))


def test_the_overview_names_all_four_optimizer_categories(overview):
    text = " ".join(messages(overview))
    for category in ("Cheapest hedge", "Strongest protection",
                     "Best efficiency", "Balanced"):
        assert category in text


def test_no_recommendation_is_called_best(overview):
    """Each category answers a different question; none of them wins outright."""
    text = " ".join(messages(overview)).lower()
    assert "best put" not in text
    assert "optimal put" not in text


def trace_values(trace, key):
    """One axis of a Plotly trace as floats.

    Plotly 6 ships arrays as base64 typed arrays rather than JSON lists, so a
    trace's x or y arrives as {"dtype": ..., "bdata": ...} and has to be
    decoded before it can be compared with anything.
    """
    import base64

    import numpy as np

    raw = trace[key]
    if isinstance(raw, dict) and "bdata" in raw:
        return np.frombuffer(base64.b64decode(raw["bdata"]),
                             dtype=np.dtype(raw["dtype"]))
    return np.asarray(raw, dtype=float)


def normalised(app):
    """Everything a viewer can read on the page, as one plain string.

    Three things get in the way of asserting on rendered text. `st.caption`
    lands in its own element list rather than in `markdown`. Components that
    style themselves emit HTML, so a bolded model name arrives wrapped in
    tags. And markdown written in a source file wraps at the column limit, so
    a sentence can be split across a newline.

    Captions are included, tags are stripped and whitespace is collapsed, so a
    test checks what is on screen rather than how it was marked up.
    """
    import re

    parts = list(messages(app))
    parts.extend(element.value for element in app.caption)
    text = re.sub(r"<[^>]+>", " ", " ".join(parts))
    return " ".join(text.split())


def chart_spec(app, index=0):
    """The rendered Plotly figure as a dict.

    Read from the element's protobuf rather than `.value`: recent Streamlit
    treats `st.plotly_chart` as a selection-capable widget, so `.value` asks
    session state for a selection that a non-interactive chart never registers.
    """
    import json

    return json.loads(app.get("plotly_chart")[index].proto.spec)


def test_the_visual_story_renders_three_panels(overview):
    spec = chart_spec(overview)
    assert len(spec["data"]) == 3          # two histograms and the payoff line
    titles = [note.get("text", "") for note in spec["layout"]["annotations"]]
    assert any("The risk" in title for title in titles)
    assert any("The instrument" in title for title in titles)
    assert any("The result" in title for title in titles)


def test_the_two_loss_panels_share_an_axis(overview):
    """Otherwise the shortened tail is an artefact of the scaling."""
    layout = chart_spec(overview)["layout"]
    assert layout["xaxis"]["range"] == layout["xaxis3"]["range"]


def test_the_visual_story_marks_both_value_at_risk_lines(overview):
    titles = [note.get("text", "")
              for note in chart_spec(overview)["layout"]["annotations"]]
    var_labels = [title for title in titles if "99% VaR" in title]
    assert len(var_labels) == 2
    assert var_labels[0] != var_labels[1]


def test_the_headline_is_read_from_the_results_not_written_in(monkeypatch):
    """Move the saved number and the page must move with it."""
    import pandas as pd

    from ui import data_loader

    real = data_loader.load_table.__wrapped__

    def shifted(name):
        frame = real(name)
        if name == "risk_percent" and frame is not None:
            frame = frame.copy()
            frame.loc[2, "cvar_99"] = 12.34
        return frame

    monkeypatch.setattr(data_loader, "load_table", shifted)
    result = AppTest.from_string(page_harness("home"), default_timeout=180).run()
    assert not result.exception
    assert any("12.34%" in value for value in messages(result))


def test_the_project_sentence_is_built_from_computed_numbers(overview):
    sentence = [value for value in messages(overview)
                if "We extend European" in value]
    assert sentence, "the one-sentence summary is missing"
    text = sentence[0]
    for fragment in ("held-out market contracts", "99% CVaR",
                     "real listed strikes"):
        assert fragment in text


def test_the_overview_never_prints_nan(overview):
    text = " ".join(messages(overview))
    assert "nan" not in text.lower().replace("finance", "")


# --------------------------------------------------------------------------
# The pricing lab (phase 5B)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lab():
    """The lab as it opens, before anything has been submitted."""
    return AppTest.from_string(page_harness("pricing_lab"),
                               default_timeout=300).run()


@pytest.fixture(scope="module")
def lab_run(lab):
    """The lab after one press of Run pricing."""
    result = AppTest.from_string(page_harness("pricing_lab"),
                                 default_timeout=300).run()
    result.button[0].click().run()
    return result


def test_the_lab_computes_nothing_until_asked(lab):
    """The whole point of the form: opening the page starts no simulation."""
    assert not lab.exception
    assert len(lab.get("plotly_chart")) == 0
    assert "pricing_lab_result" not in lab.session_state


def test_the_lab_explains_the_method_before_any_run(lab):
    assert any("How Longstaff-Schwartz works" in expander.label
               for expander in lab.expander)


def test_every_required_input_is_on_the_form(lab):
    labels = [element.label for element in lab.number_input]
    for expected in ("Spot price", "Strike", "Time to expiry (years)",
                     "Risk-free rate", "Dividend yield", "Volatility",
                     "LSMC paths", "Exercise steps", "Polynomial degree",
                     "Random seed"):
        assert expected in labels, f"{expected} is missing from the form"


def test_running_produces_the_four_charts_and_the_tabs(lab_run):
    assert not lab_run.exception
    assert len(lab_run.get("plotly_chart")) == 5   # four required, plus stops
    assert len(lab_run.tabs) == 4
    assert len(lab_run.dataframe) == 3


def test_the_headline_metrics_are_all_present(lab_run):
    text = " ".join(messages(lab_run))
    for metric in ("LSMC price", "CRR price", "Difference", "Standard error",
                   "Early exercise premium"):
        assert metric in text


def test_the_verdict_is_computed_from_the_run(lab_run):
    """The sentence has to carry the number, not a canned phrase."""
    result = lab_run.session_state["pricing_lab_result"]
    verdicts = [value for value in messages(lab_run)
                if "independent numerical methods" in value
                or "The two methods differ" in value]
    assert verdicts
    assert f"{result['relative_difference']:.2f}%" in verdicts[0]


def fresh_lab_run():
    """A lab of its own, so a test that changes an input cannot affect another."""
    result = AppTest.from_string(page_harness("pricing_lab"),
                                 default_timeout=300).run()
    result.button[0].click().run()
    return result


def paths_field(app):
    return [element for element in app.number_input
            if element.label == "LSMC paths"][0]


def test_changing_an_input_without_submitting_keeps_the_previous_result():
    """A slider must never launch fifty thousand simulations by itself."""
    app = fresh_lab_run()
    before_params = app.session_state["pricing_lab_params"]
    before_price = app.session_state["pricing_lab_result"]["lsmc_price"]

    paths_field(app).set_value(before_params.n_paths * 5).run()

    after_params = app.session_state["pricing_lab_params"]
    after_price = app.session_state["pricing_lab_result"]["lsmc_price"]
    assert after_params.n_paths == before_params.n_paths
    assert after_price == before_price
    assert not app.exception


def test_submitting_the_changed_input_does_recompute():
    """The mirror image: pressing the button must actually change the answer."""
    app = fresh_lab_run()
    before_params = app.session_state["pricing_lab_params"]
    before_price = app.session_state["pricing_lab_result"]["lsmc_price"]

    paths_field(app).set_value(2_000).run()
    app.button[0].click().run()

    after_params = app.session_state["pricing_lab_params"]
    after_price = app.session_state["pricing_lab_result"]["lsmc_price"]
    assert after_params.n_paths == 2_000 != before_params.n_paths
    assert after_price != before_price


def test_the_lab_states_its_assumptions(lab_run):
    labels = [expander.label for expander in lab_run.expander]
    assert "Model assumptions" in labels


# --------------------------------------------------------------------------
# The hedge optimizer (phase 5C)
# --------------------------------------------------------------------------

def fresh_optimizer(analyze=True):
    app = AppTest.from_string(page_harness("hedge_optimizer"),
                              default_timeout=300).run()
    if analyze:
        app.button[0].click().run()
    return app


def recommended_strike(app):
    """The strike on the recommendation card."""
    import re

    for value in messages(app):
        found = re.search(r">K=(\d+)<", value)
        if found:
            return int(found.group(1))
    return None


@pytest.fixture(scope="module")
def optimizer():
    return fresh_optimizer(analyze=False)


@pytest.fixture(scope="module")
def optimizer_run():
    return fresh_optimizer()


def test_the_optimizer_analyses_nothing_until_asked(optimizer):
    assert not optimizer.exception
    assert len(optimizer.get("plotly_chart")) == 0
    assert "hedge_table" not in optimizer.session_state


def test_every_control_the_scope_asks_for_is_present(optimizer):
    labels = [element.label for element in optimizer.selectbox]
    assert labels == ["Expiry", "Confidence level", "Risk model",
                      "Optimizer objective"]
    assert [element.label for element in optimizer.slider] == [
        "Protection weight", "Cost weight"]
    assert optimizer.button[0].label == "Analyze hedges"


def test_analysing_produces_the_three_charts_and_the_table(optimizer_run):
    assert not optimizer_run.exception
    assert len(optimizer_run.get("plotly_chart")) == 3
    assert len(optimizer_run.dataframe) == 1


def test_the_recommendation_carries_every_required_field(optimizer_run):
    text = " ".join(messages(optimizer_run))
    for field in ("Recommended strike", "Premium cost", "Hedge cost",
                  "CVaR reduction", "CVaR saved per $1 of premium"):
        assert field in text


def test_no_hedge_is_ever_called_the_best_put(optimizer_run):
    """Four objectives, four questions. None of them wins outright."""
    text = " ".join(messages(optimizer_run)).lower()
    assert "best put" not in text
    assert "optimal put" not in text
    assert "best for the selected objective" in text


def test_the_weights_always_sum_to_one():
    app = fresh_optimizer()
    for value in (0.0, 0.35, 1.0):
        app.slider[0].set_value(value).run()
        assert (app.session_state["hedge_protection_weight"]
                + app.session_state["hedge_cost_weight"]) == pytest.approx(1.0)
    app.slider[1].set_value(0.8).run()
    assert app.session_state["hedge_protection_weight"] == pytest.approx(0.2)


def test_the_weights_survive_a_visit_to_another_objective():
    """A hidden keyed widget loses its state, which would reset them to 0.5."""
    app = fresh_optimizer()
    app.slider[0].set_value(0.15).run()
    before = (app.session_state["hedge_protection_weight"],
              app.session_state["hedge_cost_weight"])

    app.selectbox[3].set_value("Cheapest").run()
    app.selectbox[3].set_value("Balanced").run()

    after = (app.session_state["hedge_protection_weight"],
             app.session_state["hedge_cost_weight"])
    assert after == before


def test_the_weights_move_the_recommendation_along_the_frontier():
    """All cost buys the cheapest hedge; all protection buys the strongest."""
    app = fresh_optimizer()
    app.slider[0].set_value(0.0).run()
    cheapest = recommended_strike(app)
    app.slider[0].set_value(1.0).run()
    strongest = recommended_strike(app)
    assert cheapest is not None and strongest is not None
    assert cheapest < strongest


def test_each_objective_selects_a_different_contract():
    app = fresh_optimizer()
    picks = {}
    for objective in ("Cheapest", "Strongest protection", "Best efficiency"):
        app.selectbox[3].set_value(objective).run()
        picks[objective] = recommended_strike(app)
    assert picks["Cheapest"] < picks["Best efficiency"] < picks["Strongest protection"]


def test_changing_the_objective_needs_no_reanalysis():
    """Re-scoring is cheap; only the risk model costs a simulation."""
    app = fresh_optimizer()
    before = app.session_state["hedge_table"]
    app.selectbox[3].set_value("Cheapest").run()
    assert app.session_state["hedge_table"] is before


def test_switching_the_risk_model_says_the_numbers_are_stale():
    app = fresh_optimizer()
    app.selectbox[2].set_value("Historical bootstrap").run()
    assert any("Showing the previous risk model" in value
               for value in messages(app))


def test_reanalysing_under_the_bootstrap_changes_the_numbers():
    app = fresh_optimizer()
    gbm = app.session_state["hedge_table"]["candidates"]["cvar_99_reduction"].tolist()
    app.selectbox[2].set_value("Historical bootstrap").run()
    app.button[0].click().run()

    payload = app.session_state["hedge_table"]
    assert payload["risk_model"] == "Historical bootstrap"
    assert payload["candidates"]["cvar_99_reduction"].tolist() != gbm


def test_the_frontier_marks_the_selected_recommendation(optimizer_run):
    spec = chart_spec(optimizer_run, 0)
    names = [trace.get("name", "") for trace in spec["data"]]
    assert "Pareto frontier" in names
    assert "Pareto efficient" in names
    assert "best for the selected objective" in names


def test_the_frontier_hover_carries_the_contract_details(optimizer_run):
    spec = chart_spec(optimizer_run, 0)
    efficient = [t for t in spec["data"] if t.get("name") == "Pareto efficient"][0]
    template = efficient["hovertemplate"]
    for field in ("Strike", "Moneyness", "Bid", "Ask", "Premium",
                  "CVaR reduction", "Efficiency"):
        assert field in template


def test_the_page_explains_how_to_read_itself(optimizer_run):
    labels = [expander.label for expander in optimizer_run.expander]
    assert "How to read this page" in labels
    assert "Model assumptions" in labels


# --------------------------------------------------------------------------
# The market validation page (phase 5D)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def validation():
    return AppTest.from_string(page_harness("market_validation"),
                               default_timeout=300).run()


def test_the_validation_page_renders(validation):
    assert not validation.exception, [e.message for e in validation.exception]


def test_it_reads_saved_results_and_needs_no_button(validation):
    """A batch result, not a calculation: it should be on screen immediately."""
    assert len(validation.get("plotly_chart")) >= 6
    assert len(validation.button) == 0


def test_every_section_the_scope_asks_for_is_present(validation):
    text = normalised(validation)
    for section in ("Which contracts were used for what",
                    "volatility curve, fitted to calibration quotes only",
                    "Predicted price against the quote",
                    "Where the error sits"):
        assert section in text


def test_the_counts_match_the_saved_tables(validation):
    import pandas as pd

    import config

    calibration = pd.read_csv(config.TABLES_DIR / "calibration_set.csv")
    heldout = pd.read_csv(config.TABLES_DIR / "heldout_predictions.csv")
    ok = heldout[heldout["prediction_status"] == "ok"]

    text = " ".join(messages(validation))
    assert f">{len(calibration)}<" in text
    assert f">{len(ok)}<" in text


def test_the_page_shows_the_two_sets_do_not_overlap(validation):
    assert any("0 in both" in value for value in messages(validation))


def test_the_error_tabs_are_all_there(validation):
    assert len(validation.tabs) == 5


def test_the_fairness_explanation_is_open_by_default(validation):
    fairness = [expander for expander in validation.expander
                if expander.label == "Why is this a fairer validation?"]
    assert fairness, "the viva explanation is missing"


def test_the_fairness_explanation_names_the_circular_test(validation):
    """It has to say what the bad version is, or it explains nothing."""
    text = normalised(validation)
    assert "circular" in text.lower()
    assert "never touches anything used to predict it" in text


def test_the_page_states_what_the_result_does_not_show(validation):
    text = normalised(validation)
    assert "does **not** show the model beats the market" in text


def test_the_smile_never_plots_a_held_out_market_volatility(validation):
    """The one thing this chart must not do.

    The held-out markers carry the volatility each contract was GIVEN by the
    curve. Plotting their own implied volatilities would suggest they helped
    shape the fit, which is precisely the leak the design removes.
    """
    import numpy as np
    import pandas as pd

    import config

    spec = chart_spec(validation, 1)
    heldout = [trace for trace in spec["data"]
               if "held out" in trace.get("name", "")]
    assert len(heldout) == 1

    plotted = trace_values(heldout[0], "y") / 100.0
    saved = pd.read_csv(config.TABLES_DIR / "heldout_predictions.csv")
    saved = saved[saved["prediction_status"] == "ok"]

    np.testing.assert_allclose(np.sort(plotted),
                               np.sort(saved["interpolated_vol"].to_numpy()),
                               rtol=1e-9)
    # And they are genuinely different numbers from the chain's own IV field.
    assert not np.allclose(np.sort(plotted),
                           np.sort(saved["quoted_iv"].to_numpy()))


def test_the_smile_curve_passes_through_the_calibration_quotes(validation):
    import numpy as np

    spec = chart_spec(validation, 1)
    quotes = [trace for trace in spec["data"]
              if trace.get("name") == "calibration quotes"][0]
    curve = [trace for trace in spec["data"]
             if "PCHIP" in trace.get("name", "")][0]

    curve_x = trace_values(curve, "x")
    curve_y = trace_values(curve, "y")
    for x, y in zip(trace_values(quotes, "x"), trace_values(quotes, "y")):
        nearest = int(np.argmin(np.abs(curve_x - x)))
        assert curve_y[nearest] == pytest.approx(y, abs=0.05)


def test_the_scatter_carries_both_pricers_and_the_identity_line(validation):
    spec = chart_spec(validation, 2)
    names = [trace.get("name", "") for trace in spec["data"]]
    assert "CRR binomial" in names
    assert "LSMC" in names
    assert any("y = x" in name for name in names)


def test_the_maturity_tab_explains_itself_when_there_is_one_expiry(validation):
    import pandas as pd

    import config

    heldout = pd.read_csv(config.TABLES_DIR / "heldout_predictions.csv")
    text = normalised(validation)
    if heldout["expiry"].nunique() == 1:
        assert "Only one maturity is quoted" in text
        assert "--refresh" in text
    else:
        assert "days to expiry" in text


def test_the_density_study_reports_that_finer_is_not_better(validation):
    text = normalised(validation)
    assert "How far apart can the calibration strikes be?" in text
    assert "Densest is not best" in text


# --------------------------------------------------------------------------
# The risk and stress lab (phase 5E)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def risk_lab():
    return AppTest.from_string(page_harness("risk_lab"),
                               default_timeout=300).run()


def test_the_risk_lab_renders(risk_lab):
    assert not risk_lab.exception, [e.message for e in risk_lab.exception]


def test_it_has_the_four_tabs_the_scope_names(risk_lab):
    assert len(risk_lab.tabs) == 4


def test_it_needs_no_button_because_it_reads_saved_runs(risk_lab):
    assert len(risk_lab.button) == 0
    assert len(risk_lab.get("plotly_chart")) >= 5


def test_both_portfolios_get_all_four_measures(risk_lab):
    text = normalised(risk_lab)
    for level in ("95%", "99%"):
        for stat in ("VaR", "CVaR"):
            assert f"{level} {stat}" in text
    assert "Unhedged: 100 SPY shares" in text
    assert "the same shares plus one put" in text


def test_the_loss_chart_draws_both_portfolios_and_marks_the_thresholds(risk_lab):
    spec = chart_spec(risk_lab, 0)
    names = [trace.get("name", "") for trace in spec["data"]]
    assert "SPY only" in names and "SPY + put" in names
    labels = [note.get("text", "") for note in spec["layout"]["annotations"]]
    assert any("95% VaR" in label for label in labels)
    assert any("99% VaR" in label for label in labels)


def test_the_protected_distribution_stops_short_of_the_unhedged_one(risk_lab):
    """The truncated tail is the finding; it has to survive into the chart."""
    spec = chart_spec(risk_lab, 0)
    traces = {trace["name"]: trace for trace in spec["data"]
              if trace.get("name") in ("SPY only", "SPY + put")}
    unhedged_counts = trace_values(traces["SPY only"], "y")
    protected_counts = trace_values(traces["SPY + put"], "y")
    centres = trace_values(traces["SPY only"], "x")

    worst_unhedged = centres[unhedged_counts > 0].max()
    worst_protected = centres[protected_counts > 0].max()
    assert worst_protected < worst_unhedged


def test_the_bootstrap_tab_shows_the_returns_it_resamples(risk_lab):
    text = normalised(risk_lab)
    assert "The returns these scenarios are drawn from" in text
    assert "Excess kurtosis" in text
    assert "Nothing is fitted here" in text


def test_the_bootstrap_tab_admits_its_limitation(risk_lab):
    """Independent draws discard volatility clustering, and that is stated."""
    assert "volatility clustering" in normalised(risk_lab)


def test_the_stress_tab_covers_every_required_shock(risk_lab):
    spec = chart_spec(risk_lab, 3)
    stock = [trace for trace in spec["data"]
             if trace.get("name") == "SPY only"][0]
    shocks = sorted(trace_values(stock, "x").tolist())
    assert shocks == pytest.approx([-30.0, -20.0, -10.0, -5.0, 0.0])


def test_the_stress_chart_shows_the_put_value_as_the_crash_deepens(risk_lab):
    spec = chart_spec(risk_lab, 3)
    puts = [trace for trace in spec["data"]
            if trace.get("name") == "put value"][0]
    values = trace_values(puts, "y")
    # x runs from the deepest shock upward, so the value must fall.
    assert values[0] > values[-1]


def test_the_protected_loss_flattens_while_the_unhedged_one_does_not(risk_lab):
    spec = chart_spec(risk_lab, 3)
    traces = {trace["name"]: trace for trace in spec["data"]
              if trace.get("name") in ("SPY only", "SPY + put")}
    stock = trace_values(traces["SPY only"], "y")
    protected = trace_values(traces["SPY + put"], "y")
    assert stock.max() - stock.min() > 25.0        # tracks the shock
    assert protected.max() - protected.min() < 6.0  # capped by the strike


def test_the_comparison_names_whichever_model_is_harsher(risk_lab):
    import pandas as pd

    import config

    frame = pd.read_csv(config.TABLES_DIR / "risk_model_comparison.csv")
    unhedged = frame[frame["put_cost_basis"] == "unhedged"]
    gbm = float(unhedged[unhedged["risk_model"] == "GBM Monte Carlo"]
                ["cvar_99_dollars"].iloc[0])
    boot = float(unhedged[unhedged["risk_model"] == "historical bootstrap"]
                 ["cvar_99_dollars"].iloc[0])
    expected = ("historical bootstrap" if boot > gbm else "GBM Monte Carlo")

    text = normalised(risk_lab)
    assert "more conservative 99% CVaR in this snapshot is produced by" in text
    assert expected in text


def test_the_harsher_model_is_read_off_the_data_not_written_in(monkeypatch):
    """Flip the saved numbers and the sentence must flip with them."""
    import pandas as pd

    from ui import data_loader

    real = data_loader.load_table.__wrapped__

    def flipped(name):
        frame = real(name)
        if name == "risk_models" and frame is not None:
            frame = frame.copy()
            mask = frame["risk_model"] == "GBM Monte Carlo"
            for column in [c for c in frame.columns if c.endswith("_dollars")]:
                frame.loc[mask, column] *= 3.0
        return frame

    monkeypatch.setattr(data_loader, "load_table", flipped)
    result = AppTest.from_string(page_harness("risk_lab"),
                                 default_timeout=300).run()
    assert not result.exception
    text = normalised(result)
    assert "produced by the GBM Monte Carlo" in text


def test_the_page_teaches_the_difference_between_var_and_cvar(risk_lab):
    text = normalised(risk_lab)
    assert "where does the bad tail begin?" in text.lower()
    assert "how bad is it on average?" in text.lower()
    assert "Expected Shortfall" in text


def test_the_comparison_explains_why_the_two_engines_differ(risk_lab):
    text = normalised(risk_lab)
    assert "share a mean and a standard deviation" in text
    labels = [expander.label for expander in risk_lab.expander]
    assert any("first two moments match" in label for label in labels)


# --------------------------------------------------------------------------
# The numerical methods page (phase 5F)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def methods():
    return AppTest.from_string(page_harness("numerical_methods"),
                               default_timeout=300).run()


def test_the_methods_page_renders(methods):
    assert not methods.exception, [e.message for e in methods.exception]


def test_every_section_is_present(methods):
    text = normalised(methods)
    for section in ("From the base paper to a decision",
                    "The methods, and why each one is here",
                    "Convergence: does more work buy accuracy?",
                    "Discretisation: how many exercise dates are enough?",
                    "Regression basis: is a higher degree better?",
                    "Interpolation: pricing once and reading it back",
                    "Invariants that must hold"):
        assert section in text


def test_the_pipeline_runs_from_european_monte_carlo_to_the_decision(methods):
    text = normalised(methods)
    for stage in ("European Monte Carlo", "Full GBM paths",
                  "Longstaff-Schwartz regression", "American early exercise",
                  "CRR lattice validation", "Pricing grid and interpolation",
                  "Real-world risk simulation", "VaR and CVaR",
                  "Hedge optimisation"):
        assert stage in text


def test_every_named_method_is_on_the_page(methods):
    text = normalised(methods)
    for method in ("Monte Carlo simulation", "Least-squares regression",
                   "Backward induction", "Binomial approximation",
                   "Root finding", "PCHIP interpolation",
                   "Empirical quantiles", "Bootstrap resampling"):
        assert method in text


def test_each_method_says_where_it_is_used_and_why(methods):
    text = normalised(methods)
    assert text.count("Where :") + text.count("Where:") >= 8
    assert text.count("Why :") + text.count("Why:") >= 8


def test_the_fitted_convergence_order_matches_the_saved_experiment(methods):
    import pandas as pd

    import config
    from src.replication import fit_convergence_order

    frame = pd.read_csv(config.TABLES_DIR / "experiment1_convergence.csv")
    fit = fit_convergence_order(frame["n_paths"].to_numpy(),
                                frame["rmse"].to_numpy())
    assert f"{fit['order']:.3f}" in normalised(methods)
    assert "-0.500" in normalised(methods)


def test_the_convergence_chart_carries_the_theoretical_reference(methods):
    spec = chart_spec(methods, 0)
    names = [trace.get("name", "") for trace in spec["data"]]
    assert any("theoretical" in name for name in names)
    assert "CRR binomial" in names


def test_the_error_band_narrows_as_the_sample_grows(methods):
    """The visible half of "does more work buy accuracy"."""
    import pandas as pd

    import config

    frame = pd.read_csv(config.TABLES_DIR / "experiment1_convergence.csv")
    frame = frame.sort_values("n_paths")
    assert frame["std_price"].iloc[-1] < frame["std_price"].iloc[0]
    assert frame["rmse"].iloc[-1] < frame["rmse"].iloc[0]


def test_the_discretisation_chart_separates_the_two_error_sources(methods):
    spec = chart_spec(methods, 1)
    names = [trace.get("name", "") for trace in spec["data"]]
    assert "Monte Carlo error" in names
    assert "discretisation error" in names


def test_the_page_explains_why_the_two_errors_are_kept_apart(methods):
    text = normalised(methods)
    assert "Added together they can cancel" in text


def test_the_regression_section_refuses_to_say_higher_is_better(methods):
    text = normalised(methods)
    assert "More complexity is not automatically better" in text
    assert f"degree {config_grid_degree()}" in text


def config_grid_degree():
    import config

    return config.GRID_DEGREE


def test_the_interpolation_section_names_the_nested_simulation_it_avoids(
        methods):
    text = normalised(methods)
    assert "nested Monte Carlo" in text


def test_every_invariant_is_derived_and_traceable(methods):
    from ui import compute

    checks = compute.sanity_checks()
    assert len(checks) >= 10
    text = normalised(methods)
    for check in checks:
        assert check["name"] in text
        assert check["source"].split(",")[0].strip() in text


def test_the_invariants_currently_all_hold(methods):
    from ui import compute

    checks = compute.sanity_checks()
    failed = [check["name"] for check in checks if not check["passed"]]
    assert not failed, f"invariants broken: {failed}"
    assert f"{len(checks)} of {len(checks)} invariants hold" in normalised(methods)


def test_the_invariants_are_recomputed_rather_than_recalled(monkeypatch,
                                                            tmp_path):
    """Move the saved results and the verdict must change with them.

    A stored pass/fail list would keep saying "pass" long after the numbers
    stopped supporting it.
    """
    import shutil

    import config
    from ui import compute

    original = compute.sanity_checks()
    assert any(check["passed"] for check in original)

    broken = tmp_path / "tables"
    shutil.copytree(config.TABLES_DIR, broken)
    frame = pd.read_csv(broken / "risk_var_cvar.csv")
    frame.loc[0, "cvar_99"] = 1.0          # below its own VaR
    frame.to_csv(broken / "risk_var_cvar.csv", index=False)

    monkeypatch.setattr(config, "TABLES_DIR", broken)
    compute.sanity_checks.clear()
    try:
        after = compute.sanity_checks()
        cvar = [check for check in after
                if check["name"] == "CVaR is never below VaR"][0]
        assert not cvar["passed"]
    finally:
        compute.sanity_checks.clear()


def test_a_failing_invariant_is_reported_in_words_not_only_colour(methods):
    """A projector washes green and red into the same grey."""
    text = normalised(methods)
    assert "PASS" in text


# --------------------------------------------------------------------------
# The methodology and about page (phase 5G)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def about():
    return AppTest.from_string(page_harness("about"), default_timeout=300).run()


def test_the_about_page_renders(about):
    assert not about.exception, [e.message for e in about.exception]


def test_it_names_the_project_and_the_course(about):
    text = normalised(about)
    assert "Extending OptionMC" in text
    assert "Least-Squares Monte Carlo" in text
    assert "CSE402 Numerical Analysis, Simulation and Modeling" in text


def test_all_three_references_are_credited(about):
    text = normalised(about)
    assert "OptionMC: A Python Package for Monte Carlo Pricing" in text
    assert "Valuing American Options by Simulation" in text
    assert "Longstaff and Schwartz, 2001" in text
    assert "Optimization of Conditional Value-at-Risk" in text
    assert "Rockafellar and Uryasev, 2000" in text


def test_it_explains_why_the_underlying_is_an_etf(about):
    """SPX and XSP are European, which would make the whole project pointless."""
    text = normalised(about)
    assert "American-style" in text
    assert "SPX" in text and "XSP" in text


def test_the_data_sources_are_named(about):
    text = normalised(about)
    for source in ("Yahoo Finance", "FRED", "DGS3MO"):
        assert source in text


def test_the_parameters_come_from_config_not_from_prose(about):
    """A number typed here would drift the moment config changed."""
    import config

    text = normalised(about)
    assert f"{config.N_RISK_SCENARIOS:,}" in text
    assert f"{config.RISK_HORIZON_DAYS} trading days" in text
    assert f"{config.LSMC_N_PATHS:,}" in text
    assert f"{config.GRID_N_POINTS} nodes" in text
    assert config.INTERPOLATION_METHOD in text
    assert str(config.SEED) in text


def test_the_measure_separation_is_stated_as_an_assumption(about):
    text = normalised(about)
    assert "risk-neutral drift r minus q" in text
    assert "historical drift mu" in text
    assert "These never mix" in text


def test_it_admits_the_bermudan_approximation(about):
    text = normalised(about)
    assert "formally Bermudan" in text
    assert "under-estimate" in text


def test_the_reproducibility_commands_are_present(about):
    assert len(about.code) == 1
    commands = about.code[0].value
    for command in ("-m pytest", "main.py --skip-fetch", "streamlit run app.py"):
        assert command in commands


def test_the_documented_paths_are_not_malformed(about):
    """A stray space in `..\\ .venv` would make every command fail."""
    commands = about.code[0].value
    assert ".. \\" not in commands
    assert "..\\ " not in commands
    assert commands.count("..\\.venv\\Scripts\\") == 4


def test_the_documented_commands_point_at_things_that_exist(about):
    """Verified against the repository, not merely spell-checked."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    assert (root.parent / ".venv" / "Scripts" / "python.exe").exists()
    assert (root.parent / ".venv" / "Scripts" / "streamlit.exe").exists()

    commands = about.code[0].value
    for script in ("main.py", "app.py"):
        assert script in commands
        assert (root / script).exists()


def test_the_offline_promise_is_made_explicitly(about):
    text = normalised(about)
    assert "Presentation mode works without live internet" in text
    assert "a failed fetch leaves the cached numbers in place" in text


def test_the_counts_are_derived_from_the_repository(about):
    """Parsed, not written down: they have to match a fresh parse."""
    from ui import compute

    facts = compute.project_facts()
    text = normalised(about)
    assert facts["test_functions"] > 100
    assert f"{facts['test_functions']:,}" in text
    assert f"{facts['tables']} tables" in text
    assert "Counted by parsing the repository, not written down here" in text


def test_the_test_count_is_honest_about_what_it_counts(about):
    """Parametrised functions expand, and the page says so."""
    text = normalised(about)
    assert "counts test *functions*" in text
    assert "expand into several cases" in text


def test_the_limitations_are_listed(about):
    text = normalised(about)
    assert "What this does not claim" in text
    assert "volatility clustering" in text
    assert "across strikes rather than across maturities" in text


def test_it_says_plainly_that_it_is_not_advice(about):
    text = normalised(about)
    assert "educational numerical-analysis project" in text
    assert "not financial advice" in text
    assert "not a recommendation to trade" in text


def test_the_page_stays_short(about):
    """The scope asks for an about page, not a second copy of the report."""
    assert len(about.markdown) < 80
    assert len(about.dataframe) == 0
