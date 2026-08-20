"""Portfolio construction, loss accounting and the real-world risk simulation."""
import numpy as np
import pytest

from src.portfolio import (compare, hedge_coverage, protective_put_portfolio,
                           unhedged_portfolio)
from src.risk_simulation import (horizon_in_years, scenario_summary,
                                 simulate_horizon_scenarios)
from src.sanity import check_measure_separation


SPOTS = np.array([700.0, 750.0, 768.37, 800.0, 850.0])
PUTS = np.array([52.0, 12.0, 5.0, 1.5, 0.1])       # falling as spot rises


def test_unhedged_values_and_losses():
    p = unhedged_portfolio(100, 768.37, SPOTS)

    assert p.initial_value == pytest.approx(76_837.0)
    assert np.allclose(p.horizon_values, 100 * SPOTS)
    assert np.allclose(p.losses, 100 * (768.37 - SPOTS))
    assert np.allclose(p.percent_losses, p.losses / 76_837.0)
    # A rise in the spot is a negative loss, i.e. a gain.
    assert p.losses[-1] < 0


def test_protective_put_values_follow_the_scope_formulas():
    """V_0 = 100 S_0 + 100 P_0 and V_h = 100 S_h + 100 P_h."""
    p = protective_put_portfolio(100, 768.37, SPOTS, put_price_now=13.53,
                                 horizon_put_prices=PUTS)

    assert p.initial_value == pytest.approx(100 * 768.37 + 100 * 13.53)
    assert np.allclose(p.horizon_values, 100 * SPOTS + 100 * PUTS)
    assert p.components["stock_now"] == pytest.approx(76_837.0)
    assert p.components["put_now"] == pytest.approx(1_353.0)
    assert p.components["covered_shares"] == 100


def test_hedge_reduces_the_spread_of_outcomes():
    a = unhedged_portfolio(100, 768.37, SPOTS)
    b = protective_put_portfolio(100, 768.37, SPOTS, 13.53, PUTS)
    assert b.percent_losses.max() < a.percent_losses.max()


def test_percentage_loss_is_measured_against_each_portfolio_own_base():
    """The protected portfolio starts out worth more, so percentages differ."""
    b = protective_put_portfolio(100, 768.37, SPOTS, 13.53, PUTS)
    assert np.allclose(b.percent_losses, b.losses / b.initial_value)
    assert b.initial_value > 100 * 768.37


def test_multiple_contracts_scale_the_hedge():
    one = protective_put_portfolio(200, 768.37, SPOTS, 13.53, PUTS, contracts=1)
    two = protective_put_portfolio(200, 768.37, SPOTS, 13.53, PUTS, contracts=2)

    assert two.components["put_now"] == pytest.approx(2 * one.components["put_now"])
    assert hedge_coverage(200, 1) == pytest.approx(0.5)
    assert hedge_coverage(200, 2) == pytest.approx(1.0)
    assert hedge_coverage(100, 1) == pytest.approx(1.0)


def test_summary_reports_the_loss_distribution():
    p = unhedged_portfolio(100, 768.37, SPOTS)
    s = p.summary()

    assert s["n_scenarios"] == 5
    assert s["mean_loss"] == pytest.approx(p.losses.mean())
    assert s["max_loss"] == pytest.approx(p.losses.max())
    assert s["probability_of_loss"] == pytest.approx(
        float((p.losses > 0).mean()))


def test_compare_returns_one_row_per_portfolio():
    a = unhedged_portfolio(100, 768.37, SPOTS)
    b = protective_put_portfolio(100, 768.37, SPOTS, 13.53, PUTS)
    frame = compare(a, b)
    assert list(frame["portfolio"]) == ["SPY only", "SPY + put"]
    assert len(frame) == 2


def test_portfolio_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        unhedged_portfolio(0, 768.37, SPOTS)
    with pytest.raises(ValueError):
        protective_put_portfolio(100, 768.37, SPOTS, -1.0, PUTS)
    with pytest.raises(ValueError, match="must match"):
        protective_put_portfolio(100, 768.37, SPOTS, 13.53, PUTS[:3])
    with pytest.raises(ValueError, match="cannot be negative"):
        protective_put_portfolio(100, 768.37, SPOTS, 13.53,
                                 np.array([-1.0, 1, 1, 1, 1]))
    with pytest.raises(ValueError):
        protective_put_portfolio(100, 768.37, SPOTS, 13.53, PUTS, contracts=0)


def test_horizon_conversion():
    assert horizon_in_years(10, 252) == pytest.approx(10 / 252)
    assert horizon_in_years(252, 252) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        horizon_in_years(0)


def test_scenarios_match_the_theoretical_moments():
    """E[S_h] = S0 exp(mu h) and the log return has std sigma sqrt(h)."""
    S0, mu, sigma = 768.37, 0.1124, 0.1721
    horizon_days = 10
    h = horizon_in_years(horizon_days)

    spots = simulate_horizon_scenarios(S0, mu, sigma, horizon_days,
                                       200_000, seed=1, antithetic=True)
    assert spots.mean() == pytest.approx(S0 * np.exp(mu * h), rel=0.002)
    log_returns = np.log(spots / S0)
    assert log_returns.std(ddof=1) == pytest.approx(sigma * np.sqrt(h), rel=0.02)
    assert np.all(spots > 0)


def test_scenarios_are_reproducible_and_seed_sensitive():
    args = (768.37, 0.1124, 0.1721, 10, 1_000)
    assert np.array_equal(simulate_horizon_scenarios(*args, seed=5),
                          simulate_horizon_scenarios(*args, seed=5))
    assert not np.array_equal(simulate_horizon_scenarios(*args, seed=5),
                              simulate_horizon_scenarios(*args, seed=6))


def test_a_higher_drift_shifts_the_scenarios_up():
    low = simulate_horizon_scenarios(768.37, 0.02, 0.1721, 10, 100_000, seed=3)
    high = simulate_horizon_scenarios(768.37, 0.20, 0.1721, 10, 100_000, seed=3)
    assert high.mean() > low.mean()


def test_scenario_summary_statistics():
    spots = simulate_horizon_scenarios(768.37, 0.1124, 0.1721, 10, 50_000,
                                       seed=7)
    s = scenario_summary(spots, 768.37)
    assert s["n_scenarios"] == 50_000
    assert s["p01_spot"] < s["p05_spot"] < s["median_spot"] < s["p95_spot"] < s["p99_spot"]
    assert s["min_spot"] <= s["p01_spot"]
    assert s["max_spot"] >= s["p99_spot"]


def test_measure_separation_guard_catches_the_classic_mistake():
    """Passing the risk-free rate as the stock drift must be flagged."""
    r, q = 0.037874, 0.0121

    assert check_measure_separation(0.1124, r, q).passed          # correct usage
    assert not check_measure_separation(r - q, r, q).passed       # risk-neutral drift
    assert not check_measure_separation(r, r, q).passed           # risk-free rate


def test_measure_separation_message_shows_both_drifts():
    """The message must name both drifts, so a reader can see they differ."""
    check = check_measure_separation(0.112408, 0.037874, 0.0121)
    assert "0.112408" in check.detail        # the real-world drift
    assert "0.025774" in check.detail        # r - q, the pricing drift
    assert "0.037874" in check.detail        # r itself
