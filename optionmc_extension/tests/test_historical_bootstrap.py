"""Tests for the historical bootstrap engine.

Two properties matter more than the rest. The scenarios must be built from
returns that actually occurred, not from a distribution fitted to them; and no
risk-free rate may reach this engine, because the drift here belongs to the
real world while `r` belongs to pricing. Both get direct tests rather than
trust.
"""
import inspect

import numpy as np
import pytest

from src.historical_bootstrap import (bootstrap_horizon_prices,
                                      bootstrap_horizon_returns,
                                      daily_log_returns, quantile_comparison,
                                      return_distribution_summary,
                                      tail_exceedance)

SPOT = 500.0
HORIZON = 10


@pytest.fixture
def returns():
    """A fat-tailed sample: mostly small days, occasionally a very bad one."""
    rng = np.random.default_rng(5)
    body = rng.normal(0.0004, 0.009, 900)
    tails = rng.normal(-0.002, 0.035, 100)
    return np.concatenate([body, tails])


# --------------------------------------------------------------------------
# Returns
# --------------------------------------------------------------------------

def test_log_returns_match_the_definition():
    prices = np.array([100.0, 110.0, 99.0])
    expected = np.array([np.log(1.10), np.log(99.0 / 110.0)])
    np.testing.assert_allclose(daily_log_returns(prices), expected)


def test_log_returns_reconstruct_the_price_path():
    prices = np.array([100.0, 103.0, 101.5, 108.0])
    rebuilt = prices[0] * np.exp(np.cumsum(daily_log_returns(prices)))
    np.testing.assert_allclose(rebuilt, prices[1:])


def test_log_returns_reject_impossible_prices():
    with pytest.raises(ValueError):
        daily_log_returns([100.0, -1.0])
    with pytest.raises(ValueError):
        daily_log_returns([100.0])


# --------------------------------------------------------------------------
# The sampling itself
# --------------------------------------------------------------------------

def test_every_sampled_return_is_one_that_actually_happened(returns):
    """No interpolation between observations, no fitted distribution."""
    drawn = bootstrap_horizon_returns(returns, 1, 3_000, seed=1)
    assert np.isin(np.round(drawn, 12), np.round(returns, 12)).all()


def test_the_horizon_price_compounds_exactly_ten_days(returns):
    prices = bootstrap_horizon_prices(SPOT, returns, HORIZON, 2_000, seed=2)
    summed = bootstrap_horizon_returns(returns, HORIZON, 2_000, seed=2)
    np.testing.assert_allclose(prices, SPOT * np.exp(summed))


def test_a_constant_return_series_gives_an_exact_answer():
    """Decisive on drift: if anything were added, this would not hold.

    With one value to draw from, every scenario must be S0 exp(10c) exactly.
    Any drift term, Ito correction or rate would show up here immediately.
    """
    c = -0.003
    prices = bootstrap_horizon_prices(SPOT, np.full(50, c), HORIZON, 500,
                                      seed=3)
    np.testing.assert_allclose(prices, SPOT * np.exp(HORIZON * c))


def test_the_engine_accepts_no_rate_drift_or_volatility():
    parameters = set(inspect.signature(bootstrap_horizon_prices).parameters)
    forbidden = {"r", "rate", "risk_free_rate", "drift", "mu", "sigma",
                 "volatility", "dividend_yield", "q"}
    assert not (parameters & forbidden)
    assert parameters == {"S0", "log_returns", "horizon_days", "n_scenarios",
                          "rng", "seed"}


def test_the_same_seed_gives_the_same_scenarios(returns):
    a = bootstrap_horizon_prices(SPOT, returns, HORIZON, 1_000, seed=42)
    b = bootstrap_horizon_prices(SPOT, returns, HORIZON, 1_000, seed=42)
    np.testing.assert_array_equal(a, b)


def test_a_different_seed_gives_different_scenarios(returns):
    a = bootstrap_horizon_prices(SPOT, returns, HORIZON, 1_000, seed=1)
    b = bootstrap_horizon_prices(SPOT, returns, HORIZON, 1_000, seed=2)
    assert not np.array_equal(a, b)


def test_the_horizon_moments_are_ten_times_the_daily_ones(returns):
    """Sampling with replacement means moments add across the horizon."""
    drawn = bootstrap_horizon_returns(returns, HORIZON, 200_000, seed=7)
    assert drawn.mean() == pytest.approx(HORIZON * returns.mean(), abs=5e-4)
    assert drawn.var() == pytest.approx(HORIZON * returns.var(), rel=0.05)


def test_prices_stay_positive(returns):
    prices = bootstrap_horizon_prices(SPOT, returns, HORIZON, 5_000, seed=9)
    assert np.all(prices > 0)


def test_scenarios_have_the_requested_shape(returns):
    assert bootstrap_horizon_prices(SPOT, returns, 5, 321, seed=1).shape == (321,)


def test_bad_inputs_are_rejected(returns):
    with pytest.raises(ValueError):
        bootstrap_horizon_prices(0.0, returns, HORIZON, 10, seed=1)
    with pytest.raises(ValueError):
        bootstrap_horizon_prices(SPOT, returns, 0, 10, seed=1)
    with pytest.raises(ValueError):
        bootstrap_horizon_prices(SPOT, returns, HORIZON, 0, seed=1)
    with pytest.raises(ValueError):
        bootstrap_horizon_prices(SPOT, [], HORIZON, 10, seed=1)


def test_a_supplied_generator_is_used(returns):
    a = bootstrap_horizon_prices(SPOT, returns, HORIZON, 200,
                                 rng=np.random.default_rng(11))
    b = bootstrap_horizon_prices(SPOT, returns, HORIZON, 200, seed=11)
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

def test_summary_recovers_known_moments():
    rng = np.random.default_rng(3)
    sample = rng.normal(0.001, 0.02, 300_000)
    summary = return_distribution_summary(sample, "normal")
    assert summary["mean"] == pytest.approx(0.001, abs=2e-4)
    assert summary["std"] == pytest.approx(0.02, rel=0.01)
    assert summary["skewness"] == pytest.approx(0.0, abs=0.02)
    assert summary["excess_kurtosis"] == pytest.approx(0.0, abs=0.05)


def test_summary_detects_fat_tails(returns):
    """The whole reason this engine exists has to be visible in the numbers."""
    assert return_distribution_summary(returns)["excess_kurtosis"] > 1.0


def test_summary_needs_enough_observations():
    with pytest.raises(ValueError):
        return_distribution_summary([0.1, 0.2])


def test_quantile_comparison_lines_up_two_samples():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=20_000), rng.normal(size=20_000)
    result = quantile_comparison(a, b)
    assert result["empirical"].shape == result["simulated"].shape
    assert result["probability"].shape == result["empirical"].shape
    # Two samples from the same law should track the diagonal.
    assert np.max(np.abs(result["empirical"] - result["simulated"])) < 0.2


def test_quantile_comparison_separates_a_fatter_tail():
    rng = np.random.default_rng(6)
    thin = rng.normal(0, 1, 60_000)
    fat = rng.standard_t(3, 60_000)
    result = quantile_comparison(fat, thin, probabilities=[0.001, 0.01, 0.5])
    assert result["empirical"][0] < result["simulated"][0]


def test_tail_exceedance_counts_correctly():
    sample = np.array([-0.3, -0.2, -0.1, 0.0, 0.1])
    counts = tail_exceedance(sample, [-0.25, -0.05])
    assert counts[-0.25] == pytest.approx(0.2)
    assert counts[-0.05] == pytest.approx(0.6)
