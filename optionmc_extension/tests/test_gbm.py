"""GBM path simulation checks."""
import numpy as np
import pytest

from src.gbm import (estimate_gbm_parameters, risk_neutral_drift,
                     simulate_gbm_paths, simulate_terminal_prices)


def test_paths_start_at_S0_and_have_right_shape():
    rng = np.random.default_rng(0)
    paths = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, n_steps=50,
                               n_paths=1000, rng=rng)
    assert paths.shape == (1000, 51)
    assert np.all(paths[:, 0] == 100.0)
    assert np.all(paths > 0.0)          # GBM can never go negative


def test_terminal_moments_match_theory():
    """E[S_T] = S0 e^{drift T} and Var[ln S_T] = sigma^2 T."""
    S0, drift, sigma, T = 100.0, 0.05, 0.2, 1.0
    rng = np.random.default_rng(7)
    paths = simulate_gbm_paths(S0, drift, sigma, T, n_steps=50,
                               n_paths=400_000, rng=rng, antithetic=True)
    terminal = paths[:, -1]

    expected_mean = S0 * np.exp(drift * T)
    assert terminal.mean() == pytest.approx(expected_mean, rel=0.01)

    log_returns = np.log(terminal / S0)
    assert log_returns.var(ddof=1) == pytest.approx(sigma ** 2 * T, rel=0.02)
    assert log_returns.mean() == pytest.approx(
        (drift - 0.5 * sigma ** 2) * T, abs=0.005)


def test_antithetic_shocks_cancel_exactly():
    """Antithetic pairing must make the log-increment sample mean vanish."""
    rng = np.random.default_rng(3)
    paths = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, n_steps=20,
                               n_paths=1000, rng=rng, antithetic=True)
    increments = np.diff(np.log(paths), axis=1)
    drift_term = (0.05 - 0.5 * 0.2 ** 2) * (1.0 / 20)
    assert increments.mean() == pytest.approx(drift_term, abs=1e-12)


def test_seed_makes_paths_reproducible():
    a = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 10, 100,
                           np.random.default_rng(42))
    b = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 10, 100,
                           np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_terminal_only_matches_full_path_distribution():
    """One exact step must have the same law as the last column of a path."""
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(11)
    stepped = simulate_gbm_paths(100.0, 0.07, 0.25, 10 / 252, n_steps=10,
                                 n_paths=200_000, rng=rng_a)[:, -1]
    direct = simulate_terminal_prices(100.0, 0.07, 0.25, 10 / 252,
                                      n_scenarios=200_000, rng=rng_b)
    assert direct.mean() == pytest.approx(stepped.mean(), rel=0.005)
    assert direct.std() == pytest.approx(stepped.std(), rel=0.02)


def test_risk_neutral_drift_subtracts_dividend_yield():
    assert risk_neutral_drift(0.05, 0.012) == pytest.approx(0.038)
    assert risk_neutral_drift(0.05) == pytest.approx(0.05)


def test_parameter_estimation_recovers_known_values():
    """Simulate with known mu/sigma, estimate them back from the prices."""
    mu, sigma = 0.09, 0.18
    rng = np.random.default_rng(123)
    n_days = 252 * 40
    daily = rng.normal((mu - 0.5 * sigma ** 2) / 252,
                       sigma / np.sqrt(252), n_days)
    prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(daily)]))

    est = estimate_gbm_parameters(prices)

    # estimate_gbm_parameters returns the drift of the GBM exponent.
    # The drift estimator is noisy: its standard error is sigma/sqrt(years),
    # which is 0.028 even with 40 years of daily data. Allow three of them --
    # a tighter tolerance would fail on sampling noise, not on a code defect.
    drift_std_error = sigma / np.sqrt(n_days / 252)
    assert est["mu"] == pytest.approx(mu - 0.5 * sigma ** 2,
                                      abs=3 * drift_std_error)
    # Volatility is estimated far more precisely, so it gets a tight bound.
    assert est["sigma"] == pytest.approx(sigma, rel=0.02)
    assert est["n_returns"] == n_days


def test_antithetic_requires_even_path_count():
    with pytest.raises(ValueError):
        simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 10, 999,
                           np.random.default_rng(0), antithetic=True)
