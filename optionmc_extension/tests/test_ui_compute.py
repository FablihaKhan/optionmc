"""Tests for the dashboard's cached calls into the numerical engine.

These wrappers are thin, but they are where a dashboard can quietly diverge
from the pipeline: a different default, a forgotten drift, a cache key that
ignores an input so the screen stops responding to it. Each of those gets a
test.
"""
import numpy as np
import pandas as pd
import pytest

from ui import compute
from ui.state import PricingParams

BASE = PricingParams(
    spot=100.0, strike=100.0, time_to_expiry=0.25, risk_free_rate=0.04,
    dividend_yield=0.01, volatility=0.25, n_paths=4_000, n_steps=25,
    degree=2, seed=7)


def with_(**changes):
    return PricingParams(**{**vars(BASE), **changes})


# --------------------------------------------------------------------------
# One valuation
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def result():
    return compute.price_once(BASE, binomial_steps=400)


def test_the_two_methods_agree_within_the_simulation_noise(result):
    """At these settings the right criterion is statistical, not a percentage.

    Four thousand paths over twenty-five exercise dates leaves a visible
    standard error, and the gap to the lattice is a couple of percent of the
    price while being well inside one standard error. Gating on a round
    percentage here would fail a run that is behaving exactly as it should.
    """
    assert abs(result["difference"]) <= 2.0 * result["lsmc_std_error"]


def test_the_gap_closes_as_paths_and_steps_increase():
    """The claim that matters: they converge on each other, not on a threshold.

    The coarse run sits slightly ABOVE the lattice, which is the small-sample
    foresight bias -- with few paths the regression fits its own noise and the
    exercise rule looks better than it is. More paths and more exercise dates
    remove it.
    """
    coarse = compute.price_once(with_(n_paths=4_000, n_steps=25),
                                binomial_steps=400)
    fine = compute.price_once(with_(n_paths=50_000, n_steps=100),
                              binomial_steps=400)
    assert fine["relative_difference"] < 0.5
    assert fine["relative_difference"] < coarse["relative_difference"]
    assert fine["lsmc_std_error"] < coarse["lsmc_std_error"]


def test_the_american_price_is_never_below_the_european_one(result):
    assert result["lsmc_price"] >= result["european_price"] - 1e-9
    assert result["binomial_american"] >= result["binomial_european"] - 1e-9


def test_the_price_is_never_below_intrinsic_value():
    """Deep in the money is where a floor is actually load-bearing."""
    deep = compute.price_once(with_(spot=60.0), binomial_steps=400)
    assert deep["intrinsic"] == pytest.approx(40.0)
    assert deep["lsmc_price"] >= deep["intrinsic"] - 1e-9


def test_the_early_exercise_premium_is_positive_for_a_put(result):
    assert result["early_exercise_premium"] > 0
    assert 0.0 < result["early_exercise_fraction"] <= 1.0


def test_both_engines_agree_on_the_size_of_the_early_exercise_premium(result):
    assert result["early_exercise_premium"] == pytest.approx(
        result["binomial_premium"], abs=0.15)


def test_the_boundary_has_one_entry_per_node(result):
    assert result["exercise_boundary"].size == BASE.n_steps + 1
    assert result["time_remaining"].size == BASE.n_steps + 1
    assert result["time_remaining"][0] == pytest.approx(BASE.time_to_expiry)
    assert result["time_remaining"][-1] == pytest.approx(0.0)


def test_the_boundary_never_rises_above_the_strike(result):
    boundary = result["exercise_boundary"]
    finite = boundary[np.isfinite(boundary)]
    assert finite.size > 0
    assert np.all(finite <= BASE.strike + 1e-9)


def test_a_fixed_seed_reproduces_the_run():
    first = compute.price_once(with_(seed=123), binomial_steps=400)
    second = compute.price_once(with_(seed=123), binomial_steps=400)
    assert first["lsmc_price"] == second["lsmc_price"]


def test_the_seed_actually_reaches_the_simulation():
    """A cache key that ignored the seed would make these identical."""
    first = compute.price_once(with_(seed=1), binomial_steps=400)
    second = compute.price_once(with_(seed=2), binomial_steps=400)
    assert first["lsmc_price"] != second["lsmc_price"]
    assert first["binomial_american"] == second["binomial_american"]


def test_every_input_changes_the_answer():
    """Each field must reach the engine, or the control on screen is a lie."""
    baseline = compute.price_once(BASE, binomial_steps=400)["lsmc_price"]
    for field, value in (("spot", 105.0), ("strike", 95.0),
                         ("time_to_expiry", 0.5), ("volatility", 0.35),
                         ("risk_free_rate", 0.08), ("dividend_yield", 0.05),
                         ("n_steps", 40), ("degree", 3)):
        moved = compute.price_once(with_(**{field: value}),
                                   binomial_steps=400)["lsmc_price"]
        assert moved != baseline, f"changing {field} did not move the price"


def test_a_larger_run_is_more_precise():
    small = compute.price_once(with_(n_paths=1_000), binomial_steps=400)
    large = compute.price_once(with_(n_paths=40_000), binomial_steps=400)
    assert large["lsmc_std_error"] < small["lsmc_std_error"]


# --------------------------------------------------------------------------
# Guards on an interactive run
# --------------------------------------------------------------------------

def test_path_counts_are_clamped_to_something_survivable():
    assert compute.clamp_paths(10) == 100
    assert compute.clamp_paths(5_000_000) == compute.MAX_PATHS
    assert compute.clamp_paths(12_345) == 12_345


def test_an_absurd_request_is_capped_rather_than_run():
    result = compute.price_once(with_(n_paths=10_000_000), binomial_steps=200)
    assert result["n_paths_used"] == compute.MAX_PATHS


# --------------------------------------------------------------------------
# Sample paths
# --------------------------------------------------------------------------

def test_sample_paths_start_at_spot_and_have_one_column_per_step():
    times, paths = compute.sample_paths(BASE, n_display=40)
    assert paths.shape == (40, BASE.n_steps + 1)
    assert np.allclose(paths[:, 0], BASE.spot)
    assert times[-1] == pytest.approx(BASE.time_to_expiry)


def test_sample_paths_do_not_disturb_the_valuation():
    """Opening a chart must not change the price on screen."""
    before = compute.price_once(BASE, binomial_steps=400)["lsmc_price"]
    compute.sample_paths(BASE)
    after = compute.price_once(BASE, binomial_steps=400)["lsmc_price"]
    assert before == after


def test_sample_paths_drift_is_risk_neutral():
    """The mean terminal price must grow at r - q, not at any historical mu."""
    params = with_(n_steps=50, time_to_expiry=1.0, volatility=0.05)
    _, paths = compute.sample_paths(params, n_display=400)
    expected = params.spot * np.exp(
        (params.risk_free_rate - params.dividend_yield) * params.time_to_expiry)
    assert paths[:, -1].mean() == pytest.approx(expected, rel=0.02)


# --------------------------------------------------------------------------
# Across spots
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def across():
    return compute.price_across_spots(with_(n_paths=6_000), n_points=11,
                                      binomial_steps=300)


def test_the_put_is_worth_less_as_the_spot_rises(across):
    assert across["lsmc"].is_monotonic_decreasing
    assert across["binomial"].is_monotonic_decreasing


def test_no_price_across_spots_falls_below_intrinsic(across):
    assert (across["lsmc"] >= across["intrinsic"] - 1e-6).all()
    assert (across["binomial"] >= across["intrinsic"] - 1e-6).all()


def test_the_two_curves_stay_close_across_the_whole_range(across):
    assert across["difference"].abs().max() < 0.25


# --------------------------------------------------------------------------
# Convergence sweep
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sweep():
    return compute.convergence_sweep(with_(n_paths=8_000), binomial_steps=400)


def test_the_sweep_visits_increasing_path_counts(sweep):
    assert sweep["n_paths"].is_monotonic_increasing
    assert len(sweep) >= 4


def test_the_confidence_band_narrows_as_paths_increase(sweep):
    width = sweep["upper"] - sweep["lower"]
    assert width.iloc[-1] < width.iloc[0]


def test_the_band_brackets_the_price(sweep):
    assert (sweep["lower"] <= sweep["price"]).all()
    assert (sweep["price"] <= sweep["upper"]).all()


def test_the_benchmark_is_the_same_at_every_point(sweep):
    """It does not depend on the number of simulated paths."""
    assert sweep["benchmark"].nunique() == 1


# --------------------------------------------------------------------------
# Display tables
# --------------------------------------------------------------------------

def test_the_summary_table_lists_every_input_and_output(result):
    frame = compute.summary_frame(BASE, result)
    quantities = set(frame["quantity"])
    for expected in ("Spot", "Strike", "Volatility", "Seed",
                     "LSMC American price", "CRR American price",
                     "Relative difference (%)"):
        assert expected in quantities
    assert frame["value"].notna().all()


def test_the_boundary_table_matches_the_run(result):
    frame = compute.exercise_boundary_frame(result)
    assert len(frame) == BASE.n_steps + 1
    assert frame["in_the_money_paths"].max() > 0


# --------------------------------------------------------------------------
# The hedge optimizer table
# --------------------------------------------------------------------------

import config

CANDIDATES = config.TABLES_DIR / "hedge_optimizer_candidates.csv"
needs_optimizer = pytest.mark.skipif(
    not CANDIDATES.exists(),
    reason="run experiments/hedge_optimizer_experiment.py first")


@pytest.fixture(scope="module")
def gbm_table():
    return compute.hedge_candidates_table("GBM Monte Carlo", 20_000, 10, 42)


@pytest.fixture(scope="module")
def bootstrap_table():
    return compute.hedge_candidates_table("Historical bootstrap", 20_000, 10, 42)


@needs_optimizer
def test_every_cached_candidate_is_priced(gbm_table):
    frame = gbm_table["candidates"]
    assert len(frame) == len(pd.read_csv(CANDIDATES))
    assert not gbm_table["missing_grids"]
    for column in ("cvar_95_reduction", "cvar_99_reduction", "premium_cost",
                   "hedge_cost_percent"):
        assert frame[column].notna().all()


@needs_optimizer
def test_the_premium_is_the_ask_not_the_mid(gbm_table):
    frame = gbm_table["candidates"]
    assert np.allclose(frame["premium_cost"],
                       frame["ask"] * config.CONTRACT_MULTIPLIER)
    assert (frame["ask"] >= frame["mid"]).all()


@needs_optimizer
def test_a_closer_strike_costs_more_and_protects_more(gbm_table):
    frame = gbm_table["candidates"].sort_values("strike")
    assert frame["premium_cost"].is_monotonic_increasing
    assert frame["cvar_99_reduction"].is_monotonic_increasing


@needs_optimizer
def test_cvar_is_reduced_further_than_var(gbm_table):
    """The put works hardest deep in the tail, which is what CVaR averages."""
    frame = gbm_table["candidates"]
    assert (frame["cvar_99_reduction"] > frame["var_99_reduction"]).all()


@needs_optimizer
def test_no_scenario_needed_extrapolation(gbm_table):
    assert (gbm_table["candidates"]["scenarios_outside_grid"] == 0).all()


@needs_optimizer
def test_the_bootstrap_is_harsher_on_the_unhedged_position(gbm_table,
                                                           bootstrap_table):
    assert (bootstrap_table["baseline"]["cvar_99_dollars"]
            > gbm_table["baseline"]["cvar_99_dollars"])


@needs_optimizer
def test_the_hedge_helps_under_both_risk_models(gbm_table, bootstrap_table):
    for table in (gbm_table, bootstrap_table):
        assert (table["candidates"]["cvar_99_reduction"] > 0).all()


@needs_optimizer
def test_the_risk_model_reaches_the_numbers(gbm_table, bootstrap_table):
    """A cache key that ignored the model would make these identical."""
    assert not np.allclose(gbm_table["candidates"]["cvar_99_reduction"],
                           bootstrap_table["candidates"]["cvar_99_reduction"])


@needs_optimizer
def test_missing_optimizer_results_return_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TABLES_DIR", tmp_path)
    compute.hedge_candidates_table.clear()
    try:
        assert compute.hedge_candidates_table("GBM Monte Carlo", 1_000, 10, 1) is None
    finally:
        compute.hedge_candidates_table.clear()


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------

@needs_optimizer
def test_each_objective_picks_the_row_that_wins_it(gbm_table):
    frame, winners = compute.rank_for_objective(gbm_table["candidates"],
                                                0.99, 0.5, 0.5)
    cheapest = compute.objective_winner(frame, winners, "Cheapest", 0.99)
    strongest = compute.objective_winner(frame, winners, "Strongest protection",
                                         0.99)
    efficient = compute.objective_winner(frame, winners, "Best efficiency", 0.99)

    assert cheapest["premium_cost"] == frame["premium_cost"].min()
    assert strongest["cvar_99_reduction"] == frame["cvar_99_reduction"].max()
    assert (efficient["cvar_99_saved_per_premium_dollar"]
            == frame["cvar_99_saved_per_premium_dollar"].max())


@needs_optimizer
def test_all_cost_weight_picks_the_cheapest_and_all_protection_the_strongest(
        gbm_table):
    """The slider has to actually move the answer, or it is decoration."""
    _, cost_only = compute.rank_for_objective(gbm_table["candidates"], 0.99,
                                              0.0, 1.0)
    _, protection_only = compute.rank_for_objective(gbm_table["candidates"],
                                                    0.99, 1.0, 0.0)
    assert (cost_only["balanced"]["strike"]
            < protection_only["balanced"]["strike"])
    assert cost_only["balanced"]["strike"] == cost_only["cheapest"]["strike"]
    assert (protection_only["balanced"]["strike"]
            == protection_only["strongest"]["strike"])


@needs_optimizer
def test_the_confidence_level_selects_its_own_column(gbm_table):
    at95, _ = compute.rank_for_objective(gbm_table["candidates"], 0.95, 0.5, 0.5)
    at99, _ = compute.rank_for_objective(gbm_table["candidates"], 0.99, 0.5, 0.5)
    assert not np.allclose(at95["protection_score"], at99["protection_score"])


@needs_optimizer
def test_the_frontier_holds_no_dominated_point(gbm_table):
    frame, _ = compute.rank_for_objective(gbm_table["candidates"], 0.99,
                                          0.5, 0.5)
    costs = frame["premium_cost"].to_numpy()
    protection = frame["cvar_99_reduction"].to_numpy()
    for i in np.flatnonzero(frame["pareto_efficient"].to_numpy()):
        dominating = ((costs <= costs[i]) & (protection >= protection[i])
                      & ((costs < costs[i]) | (protection > protection[i])))
        assert not dominating.any()
