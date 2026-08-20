"""Tests for the protective put optimizer.

The interesting failures here are not crashes but plausible wrong answers: a
recommendation that does not actually win its own category, a frontier that
quietly contains a dominated point, or a comparison where two candidates were
measured on different scenarios. Each of those gets its own test.
"""
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from src.binomial import crr_american_put
from src.hedge_optimizer import (DEFAULT_TARGET_MONEYNESS, baseline_risk_row,
                                 candidate_grid, cost_percent,
                                 evaluate_candidates, min_max_score,
                                 nearest_listed_strike, pareto_mask,
                                 premium_cost, rank_candidates,
                                 select_candidate_puts)
from src.market_data import MarketDataError
from src.pricing_grid import moneyness_grid
from src.risk_simulation import simulate_horizon_scenarios

SPOT = 100.0
RATE = 0.04
DIVIDEND = 0.01
MATURITY = 0.25
TRUE_SIGMA = 0.25


def make_chain(strikes=(88.0, 90.0, 92.0, 95.0, 97.0, 100.0, 103.0),
               sigma=TRUE_SIGMA, spread=0.02, bad_strikes=()):
    """A put chain whose mids are real CRR prices at a known volatility.

    Building the quotes from the pricer means the calibration step has an exact
    answer to find, so a round trip through implied volatility is a genuine
    check rather than a tautology about noise.
    """
    rows = []
    for strike in strikes:
        mid = crr_american_put(SPOT, strike, MATURITY, RATE, sigma, DIVIDEND,
                               n_steps=400)
        bid, ask = mid - spread / 2, mid + spread / 2
        if strike in bad_strikes:
            bid, ask = 0.0, 0.0
        rows.append({
            "contractSymbol": f"TEST{strike:g}P",
            "strike": strike,
            "bid": bid,
            "ask": ask,
            "lastPrice": mid,
            "impliedVolatility": sigma,
            "volume": 10.0,
            "openInterest": 100.0,
            "expiry": "2026-10-30",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def snapshot():
    return SimpleNamespace(spot=SPOT, risk_free_rate=RATE,
                           dividend_yield=DIVIDEND, time_to_expiry=MATURITY,
                           historical_drift=0.09, historical_volatility=0.20)


@pytest.fixture
def horizon_spots():
    return simulate_horizon_scenarios(
        S0=SPOT, real_world_drift=0.09, sigma=0.20, horizon_days=10,
        n_scenarios=4_000, seed=7)


def evaluate(chain, snapshot, horizon_spots, targets=(0.90, 0.95, 1.00),
             seed=42, tmp_path=None):
    candidates, _ = select_candidate_puts(chain, SPOT, targets)
    grid_spots = moneyness_grid(SPOT, 0.60, 1.40, 17)
    rows, baseline, extras = evaluate_candidates(
        candidates, snapshot, horizon_spots, MATURITY - 10 / 252, grid_spots,
        lsmc_paths=2_000, lsmc_steps=20, binomial_steps=300, grid_paths=5_000,
        grid_steps=20, grid_degree=3, seed=seed, iv_tree_steps=200,
        cache_dir=tmp_path)
    return pd.DataFrame(rows), baseline, extras


# --------------------------------------------------------------------------
# Candidate selection: real strikes, unique, nothing dropped silently
# --------------------------------------------------------------------------

def test_nearest_listed_strike_returns_a_listed_value():
    strikes = [88.0, 92.0, 97.0]
    assert nearest_listed_strike(strikes, 90.5) == 92.0
    assert nearest_listed_strike(strikes, 89.0) == 88.0


def test_every_candidate_strike_is_really_listed():
    chain = make_chain()
    candidates, _ = select_candidate_puts(chain, SPOT)
    listed = set(chain["strike"])
    assert {c.strike for c in candidates} <= listed


def test_candidate_strikes_are_unique_when_targets_collide():
    # Only three listed strikes, so five target ratios must collapse onto them.
    chain = make_chain(strikes=(90.0, 95.0, 100.0))
    candidates, _ = select_candidate_puts(chain, SPOT, DEFAULT_TARGET_MONEYNESS)
    strikes = [c.strike for c in candidates]
    assert len(strikes) == len(set(strikes))
    assert len(strikes) <= 3


def test_contract_without_a_usable_quote_is_reported_not_dropped():
    chain = make_chain(bad_strikes=(90.0,))
    candidates, rejects = select_candidate_puts(chain, SPOT, (0.90, 0.95, 1.00))
    assert 90.0 not in {c.strike for c in candidates}
    assert any(r["strike"] == 90.0 for r in rejects)
    assert rejects[0]["reason"]


def test_selection_records_the_quote_source():
    candidates, _ = select_candidate_puts(make_chain(), SPOT)
    assert {c.price_source for c in candidates} == {"mid"}


def test_empty_chain_is_rejected():
    with pytest.raises(MarketDataError):
        select_candidate_puts(pd.DataFrame({"strike": []}), SPOT)


# --------------------------------------------------------------------------
# Cost convention
# --------------------------------------------------------------------------

def test_premium_cost_uses_the_contract_multiplier():
    assert premium_cost(4.66) == pytest.approx(466.0)
    assert premium_cost(4.66, contracts=2) == pytest.approx(932.0)


def test_premium_cost_rejects_nonsense():
    with pytest.raises(ValueError):
        premium_cost(-1.0)
    with pytest.raises(ValueError):
        premium_cost(1.0, contracts=0)


def test_cost_percent_is_relative_to_the_share_position():
    assert cost_percent(466.0, 76_837.0) == pytest.approx(0.60648, abs=1e-5)
    with pytest.raises(ValueError):
        cost_percent(466.0, 0.0)


def test_acquisition_cost_uses_the_ask_not_the_mid(snapshot, horizon_spots):
    frame, _, _ = evaluate(make_chain(spread=1.0), snapshot, horizon_spots)
    assert np.allclose(frame["premium_cost"], frame["ask"] * 100)
    # With a wide spread the mid would give a visibly cheaper answer, so this
    # is not a distinction without a difference.
    assert np.all(frame["ask"] > frame["mid"])


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def test_min_max_score_spans_zero_to_one():
    assert min_max_score([1.0, 2.0, 3.0]) == pytest.approx([0.0, 0.5, 1.0])


def test_min_max_score_can_be_reversed():
    assert min_max_score([1.0, 2.0, 3.0], False) == pytest.approx([1.0, 0.5, 0.0])


def test_min_max_score_is_defined_when_every_value_is_equal():
    # Documented behaviour, not a divide-by-zero nan leaking into a ranking.
    assert min_max_score([5.0, 5.0, 5.0]) == pytest.approx([0.5, 0.5, 0.5])


def test_pareto_keeps_a_monotone_trade_off_intact():
    assert pareto_mask([1, 2, 3], [10, 20, 30]).all()


def test_pareto_excludes_a_dominated_point():
    # index 2 costs more than index 1 and protects less: dominated.
    mask = pareto_mask([1, 2, 2.5, 3], [10, 20, 15, 30])
    assert list(mask) == [True, True, False, True]


def test_no_point_on_the_frontier_is_dominated():
    rng = np.random.default_rng(3)
    costs = rng.uniform(1, 10, 40)
    protection = rng.uniform(0, 100, 40)
    mask = pareto_mask(costs, protection)
    for i in np.flatnonzero(mask):
        dominating = ((costs <= costs[i]) & (protection >= protection[i])
                      & ((costs < costs[i]) | (protection > protection[i])))
        assert not dominating.any()


def test_weights_must_sum_to_one():
    frame = pd.DataFrame({"cvar_99_reduction": [10.0, 20.0],
                          "premium_cost": [100.0, 200.0],
                          "cvar_99_saved_per_premium_dollar": [1.0, 2.0]})
    with pytest.raises(ValueError):
        rank_candidates(frame, 0.7, 0.7)
    with pytest.raises(ValueError):
        rank_candidates(frame, -0.5, 1.5)


def test_all_protection_weight_picks_the_strongest():
    frame = pd.DataFrame({"cvar_99_reduction": [10.0, 55.0, 30.0],
                          "premium_cost": [100.0, 900.0, 400.0],
                          "cvar_99_saved_per_premium_dollar": [3.0, 1.0, 2.0]})
    _, winners = rank_candidates(frame, 1.0, 0.0)
    assert winners["balanced"]["cvar_99_reduction"] == 55.0


def test_all_cost_weight_picks_the_cheapest():
    frame = pd.DataFrame({"cvar_99_reduction": [10.0, 55.0, 30.0],
                          "premium_cost": [100.0, 900.0, 400.0],
                          "cvar_99_saved_per_premium_dollar": [3.0, 1.0, 2.0]})
    _, winners = rank_candidates(frame, 0.0, 1.0)
    assert winners["balanced"]["premium_cost"] == 100.0


# --------------------------------------------------------------------------
# The recommendations must actually win their own category
# --------------------------------------------------------------------------

def test_cheapest_really_has_the_minimum_premium(snapshot, horizon_spots):
    frame, _, _ = evaluate(make_chain(), snapshot, horizon_spots)
    frame, winners = rank_candidates(frame)
    assert winners["cheapest"]["premium_cost"] == frame["premium_cost"].min()


def test_strongest_really_has_the_maximum_cvar_reduction(snapshot, horizon_spots):
    frame, _, _ = evaluate(make_chain(), snapshot, horizon_spots)
    frame, winners = rank_candidates(frame)
    assert (winners["strongest"]["cvar_99_reduction"]
            == frame["cvar_99_reduction"].max())


def test_efficiency_winner_really_maximises_efficiency(snapshot, horizon_spots):
    frame, _, _ = evaluate(make_chain(), snapshot, horizon_spots)
    frame, winners = rank_candidates(frame)
    assert (winners["most_efficient"]["cvar_99_saved_per_premium_dollar"]
            == frame["cvar_99_saved_per_premium_dollar"].max())


def test_frontier_from_a_real_evaluation_has_no_dominated_point(
        snapshot, horizon_spots):
    frame, _, _ = evaluate(make_chain(), snapshot, horizon_spots)
    frame, _ = rank_candidates(frame)
    costs = frame["premium_cost"].to_numpy()
    protection = frame["cvar_99_reduction"].to_numpy()
    for i in np.flatnonzero(frame["pareto_efficient"].to_numpy()):
        dominating = ((costs <= costs[i]) & (protection >= protection[i])
                      & ((costs < costs[i]) | (protection > protection[i])))
        assert not dominating.any()


# --------------------------------------------------------------------------
# Risk numbers
# --------------------------------------------------------------------------

def test_cvar_is_at_least_var_for_every_candidate(snapshot, horizon_spots):
    frame, baseline, _ = evaluate(make_chain(), snapshot, horizon_spots)
    for _, row in frame.iterrows():
        for key in ("95", "99"):
            assert row[f"cvar_{key}_dollars"] >= row[f"var_{key}_dollars"]
            assert row[f"cvar_{key}_percent"] >= row[f"var_{key}_percent"]
    base = baseline_risk_row(baseline)
    for key in ("95", "99"):
        assert base[f"cvar_{key}_dollars"] >= base[f"var_{key}_dollars"]


def test_a_hedge_closer_to_the_money_costs_more_and_protects_more(
        snapshot, horizon_spots):
    frame, _, _ = evaluate(make_chain(), snapshot, horizon_spots)
    frame = frame.sort_values("strike")
    assert frame["premium_cost"].is_monotonic_increasing
    assert frame["cvar_99_reduction"].is_monotonic_increasing


def test_every_candidate_is_measured_on_the_same_scenarios(
        snapshot, horizon_spots):
    """The unhedged benchmark is built once; if it were rebuilt per candidate
    the reductions would carry the noise of a different sample each time."""
    _, baseline_a, _ = evaluate(make_chain(), snapshot, horizon_spots,
                                targets=(0.90, 0.95))
    _, baseline_b, _ = evaluate(make_chain(), snapshot, horizon_spots,
                                targets=(0.95, 1.00))
    assert baseline_a.initial_value == baseline_b.initial_value
    np.testing.assert_array_equal(baseline_a.losses, baseline_b.losses)


# --------------------------------------------------------------------------
# Calibration and reproducibility
# --------------------------------------------------------------------------

def test_calibrated_sigma_reproduces_the_quote_it_came_from(
        snapshot, horizon_spots):
    """An implied volatility is implied relative to a particular tree.

    So the round trip has to be closed on the same discretisation it was opened
    on -- 200 steps here, matching `iv_tree_steps` in `evaluate`. Repricing on a
    400-step tree instead moves the answer by about half a cent, which is the
    tree's own discretisation error and not a calibration failure.
    """
    frame, _, _ = evaluate(make_chain(), snapshot, horizon_spots)
    for _, row in frame.iterrows():
        repriced = crr_american_put(SPOT, row["strike"], MATURITY, RATE,
                                    row["sigma"], DIVIDEND, n_steps=200)
        assert repriced == pytest.approx(row["market_mid"], abs=1e-4)


def test_calibration_recovers_the_volatility_the_chain_was_built_at(
        snapshot, horizon_spots):
    frame, _, _ = evaluate(make_chain(spread=0.0), snapshot, horizon_spots)
    assert frame["sigma"].to_numpy() == pytest.approx(TRUE_SIGMA, abs=2e-3)


def test_evaluation_is_reproducible_with_a_fixed_seed(snapshot, horizon_spots):
    first, _, _ = evaluate(make_chain(), snapshot, horizon_spots, seed=123)
    second, _, _ = evaluate(make_chain(), snapshot, horizon_spots, seed=123)
    numeric = first.select_dtypes(include=[np.number]).columns
    pd.testing.assert_frame_equal(first[numeric], second[numeric])


def test_a_different_seed_moves_the_estimate_but_not_the_conclusion(
        snapshot, horizon_spots):
    first, _, _ = evaluate(make_chain(), snapshot, horizon_spots, seed=1)
    second, _, _ = evaluate(make_chain(), snapshot, horizon_spots, seed=2)
    assert not np.array_equal(first["lsmc_price"], second["lsmc_price"])
    # The ordering the recommendation rests on is not seed noise.
    assert (first.sort_values("strike")["cvar_99_reduction"].is_monotonic_increasing
            and second.sort_values("strike")["cvar_99_reduction"].is_monotonic_increasing)


def test_candidate_grid_is_cached_and_reused(snapshot, horizon_spots, tmp_path):
    candidates, _ = select_candidate_puts(make_chain(), SPOT, (0.95,))
    spots = moneyness_grid(SPOT, 0.60, 1.40, 17)
    args = dict(spots=spots, t_remaining=MATURITY - 10 / 252,
                risk_free_rate=RATE, dividend_yield=DIVIDEND, n_paths=3_000,
                n_steps=20, degree=3, seed=42, cache_dir=tmp_path)

    first, source_a = candidate_grid(candidates[0], TRUE_SIGMA, **args)
    second, source_b = candidate_grid(candidates[0], TRUE_SIGMA, **args)
    assert source_a == "built" and source_b == "cached"
    np.testing.assert_allclose(first.prices, second.prices)

    # A different volatility must not silently reuse the stale curve.
    _, source_c = candidate_grid(candidates[0], TRUE_SIGMA + 0.05, **args)
    assert source_c == "built"


def test_lsmc_and_the_tree_agree_on_every_candidate(snapshot, horizon_spots):
    candidates, _ = select_candidate_puts(make_chain(), SPOT, (0.90, 0.95, 1.00))
    grid_spots = moneyness_grid(SPOT, 0.60, 1.40, 17)
    rows, _, _ = evaluate_candidates(
        candidates, snapshot, horizon_spots, MATURITY - 10 / 252, grid_spots,
        lsmc_paths=40_000, lsmc_steps=50, binomial_steps=800, grid_paths=3_000,
        grid_steps=20, seed=5, iv_tree_steps=200)
    for row in rows:
        assert row["rel_error_vs_binomial"] < 0.02
