"""Checks that the saved results survive being read back.

A CSV that writes one thing and reads back another is a quiet failure: the
experiment prints the right table, the file looks right in a text editor, and
the dashboard silently drops a section. That happened once here -- a column
written as "n/a" came back as NaN, because "n/a" is one of the tokens pandas
treats as missing by default -- so it now has a test.

These run against whatever is on disk and skip when a phase has not been run,
so they never fail on a fresh checkout.
"""
import pandas as pd
import pytest

import config

# What pandas silently turns into NaN. Writing any of these as a *label* makes
# the column unusable for selecting rows.
NA_TOKENS = {
    "-1.#IND", "1.#QNAN", "1.#IND", "-1.#QNAN", "#N/A N/A", "#N/A", "N/A",
    "n/a", "NA", "<NA>", "#NA", "NULL", "null", "NaN", "-NaN", "nan", "-nan",
    "None", "none",
}

TABLES = sorted(config.TABLES_DIR.glob("*.csv"))


def existing(name):
    path = config.TABLES_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not on disk; run the pipeline first")
    return path


@pytest.mark.skipif(not TABLES, reason="no result tables on disk")
@pytest.mark.parametrize("path", TABLES, ids=lambda p: p.name)
def test_no_label_is_written_as_a_pandas_missing_token(path):
    """Reading a table back must give the values the experiment wrote.

    Compared against a read with the NA machinery switched off, so the test
    sees the literal text in the file rather than pandas' interpretation of it.
    """
    interpreted = pd.read_csv(path)
    literal = pd.read_csv(path, keep_default_na=False, dtype=str)

    offenders = []
    for column in literal.columns:
        raw = literal[column]
        lost = raw[raw.isin(NA_TOKENS)]
        if len(lost):
            offenders.append(f"{column}: {sorted(set(lost))[:3]}")

    assert not offenders, (
        f"{path.name} writes labels pandas reads back as NaN: "
        + "; ".join(offenders))


def test_the_risk_model_table_can_still_find_its_unhedged_rows():
    """The exact regression: the dashboard selects on this column."""
    frame = pd.read_csv(existing("risk_model_comparison.csv"))
    unhedged = frame[frame["put_cost_basis"] == "unhedged"]
    assert len(unhedged) == frame["risk_model"].nunique()
    assert frame["put_cost_basis"].notna().all()


def test_every_risk_model_reports_all_four_measures():
    frame = pd.read_csv(existing("risk_model_comparison.csv"))
    for column in ("var_95_dollars", "cvar_95_dollars",
                   "var_99_dollars", "cvar_99_dollars"):
        assert frame[column].notna().all()
        assert (frame[column] > 0).all()


def test_cvar_is_never_below_var_in_any_saved_table():
    """A saved result that broke this would be a corrupted file, not a finding."""
    frame = pd.read_csv(existing("risk_model_comparison.csv"))
    for level in ("95", "99"):
        assert (frame[f"cvar_{level}_dollars"]
                >= frame[f"var_{level}_dollars"]).all()


def test_the_ranking_table_names_all_four_categories():
    frame = pd.read_csv(existing("hedge_optimizer_rankings.csv"))
    assert set(frame["category"]) == {"cheapest", "strongest",
                                      "most_efficient", "balanced"}


def test_the_heldout_predictions_are_all_usable():
    frame = pd.read_csv(existing("heldout_predictions.csv"))
    ok = frame[frame["prediction_status"] == "ok"]
    assert len(ok) > 0
    for column in ("interpolated_vol", "crr_price", "lsmc_price", "mid"):
        assert ok[column].notna().all()
        assert (ok[column] > 0).all()


def test_the_stress_table_covers_every_required_shock():
    frame = pd.read_csv(existing("stress_test_results.csv"))
    for shock in (0.0, -0.05, -0.10, -0.20, -0.30):
        assert (frame["shock"] - shock).abs().min() < 1e-9


@pytest.mark.skipif(not TABLES, reason="no result tables on disk")
@pytest.mark.parametrize("path", TABLES, ids=lambda p: p.name)
def test_no_saved_table_is_empty(path):
    assert len(pd.read_csv(path)) > 0
